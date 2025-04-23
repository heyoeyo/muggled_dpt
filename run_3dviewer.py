#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# ---------------------------------------------------------------------------------------------------------------------
# %% Imports

import os
import os.path as osp
import json
import argparse
import webbrowser
import socket
from dataclasses import dataclass
from http.server import HTTPServer, BaseHTTPRequestHandler

import cv2
import numpy as np
import torch
import torch.nn as nn

from lib.make_dpt import make_dpt_from_state_dict

from lib.demo_helpers.history_keeper import HistoryKeeper
from lib.demo_helpers.loading import ask_for_path_if_missing, ask_for_model_path_if_missing
from lib.demo_helpers.misc import get_default_device_string, make_device_config
from lib.demo_helpers.video import create_video_capture


# ---------------------------------------------------------------------------------------------------------------------
# %% Handle script args

# Set argparse defaults
default_device = get_default_device_string()
default_image_path = None
default_model_path = None
default_base_size = None
default_server_host = "localhost"
default_server_port = 5678
default_image_encoding = "jpg"
default_depth_encoding = "png"

# Define script arguments
parser = argparse.ArgumentParser(description="Launches a server for a web-based 3D viewer of depth predictions")
parser.add_argument("-i", "--input_path", default=default_image_path, help="Path to input image or video")
parser.add_argument("-m", "--model_path", default=default_model_path, type=str, help="Path to model weights")
parser.add_argument(
    "-k",
    "--mask_path",
    default=None,
    type=str,
    help="Path to a binary mask image, which can be used to eliminate sections outside of masked areas",
)
parser.add_argument(
    "-d",
    "--device",
    default=default_device,
    type=str,
    help="Device to use when running model (ex: 'cpu', 'cuda', 'mps')",
)
parser.add_argument(
    "-nc", "--no_cache", default=False, action="store_true", help="Disable caching to reduce VRAM usage"
)
parser.add_argument(
    "-f16",
    "--use_float16",
    default=False,
    action="store_true",
    help="Use 16-bit floating point model weights. This reduces prediction quality, but will run faster",
)
parser.add_argument(
    "-z",
    "--no_optimization",
    default=False,
    action="store_true",
    help="Disable attention optimizations (only effects DepthAnything models)",
)
parser.add_argument(
    "-ar",
    "--use_aspect_ratio",
    default=False,
    action="store_true",
    help="Process the image at it's original aspect ratio, if the model supports it",
)
parser.add_argument(
    "-b", "--base_size_px", default=default_base_size, type=int, help="Override base (e.g. 384, 512) model size"
)
parser.add_argument(
    "-t", "--host", default=default_server_host, type=str, help=f"Server host (default: {default_server_host})"
)
parser.add_argument(
    "-p", "--port", default=default_server_port, type=int, help=f"Server port (default: {default_server_port})"
)
parser.add_argument(
    "-cam",
    "--use_webcam",
    default=False,
    action="store_true",
    help="Use a webcam as the video input, instead of a file",
)
parser.add_argument(
    "-l",
    "--launch",
    default=False,
    action="store_true",
    help="Automatically launch the browser with the webpage running",
)
parser.add_argument(
    "--encode_image",
    default=default_image_encoding,
    type=str,
    help=f"Image encoding format for RGB image data (default: {default_image_encoding})",
)
parser.add_argument(
    "--encode_depth",
    default=default_depth_encoding,
    type=str,
    help=f"Image encoding format for depth data (default: {default_depth_encoding})",
)

# For convenience
args = parser.parse_args()
arg_input_path = args.input_path
arg_model_path = args.model_path
device_str = args.device
use_cache = not args.no_cache
use_float32 = not args.use_float16
use_optimizations = not args.no_optimization
force_square_resolution = not args.use_aspect_ratio
model_base_size = args.base_size_px
server_host = args.host
server_port = args.port
use_webcam = args.use_webcam
launch_browser = args.launch
arg_image_encoding = str(args.encode_image).lower()
arg_depth_encoding = str(args.encode_depth).lower()
arg_mask_path = args.mask_path

# Create history to re-use selected inputs
history = HistoryKeeper()
_, history_inputpath = history.read("input3d_path")
_, history_modelpath = history.read("model_path")

# Get pathing to resources, if not provided already
input_path = ask_for_path_if_missing(arg_input_path, "image or video", history_inputpath) if not use_webcam else 0
model_path = ask_for_model_path_if_missing(__file__, arg_model_path, history_modelpath)

# Store history for use on reload (but don't save video path when using webcam)
if use_webcam:
    history.store(model_path=model_path)
else:
    history.store(input3d_path=input_path, model_path=model_path)


# ---------------------------------------------------------------------------------------------------------------------
# %% Classes


class VideoData:
    """Class used to hold video reader/state data for handling video inputs"""

    def __init__(self, video_path: str):

        assert osp.exists(video_path), f"Error, video does not exist: {video_path}"
        self._vread: cv2.VideoCapture = create_video_capture(video_path)

        self._curr_idx: int = 0
        self._wh = (int(self._vread.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self._vread.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        self._total_frames = int(self._vread.get(cv2.CAP_PROP_FRAME_COUNT))

    def get_total_frames(self) -> int:
        return self._total_frames

    def get_wh(self) -> tuple[int, int]:
        return self._wh

    def set_frame_index(self, frame_index: int) -> None:
        self._vread.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        self._curr_idx = frame_index
        return

    def get_frame_index(self) -> int:
        return self._curr_idx

    def read_frame(self, frame_index: int = 0) -> np.ndarray:

        # Jump to frame, if it's not just the next frame
        if frame_index != self._curr_idx:
            if frame_index > 0:
                print("FRAME INDEX MISMATCH! Current:", self._curr_idx, "New:", frame_index)
            self.set_frame_index(frame_index)

        # Read frame, with 'wrap-around' if there's is no frame (which happens at the end of videos)
        ok_frame, frame = self._vread.read()
        if not ok_frame:
            self.set_frame_index(0)
            re_ok_frame, frame = self._vread.read()
            assert re_ok_frame, f"Error, cannot read frame {frame_index} from video!"

        # Update record of which frame we're on
        self._curr_idx += 1

        return frame

    def close(self):
        self._vread.release()


class ImageAsVideoData:
    """Class used to 'fake' a video source when using a static image"""

    def __init__(self, image_path: str, **kwargs):

        assert osp.exists(image_path), f"Error, image does not exist: {image_path}"
        self._image: np.ndarray = cv2.imread(image_path)
        assert self._image is not None, f"Error reading image: {image_path}"

    def get_total_frames(self) -> int:
        return 1

    def get_wh(self) -> tuple[int, int]:
        return (self._image.shape[1], self._image.shape[0])

    def set_frame_index(self, frame_index: int) -> None:
        # Nothing to set for image as a video
        return

    def get_frame_index(self) -> int:
        return 0

    def read_frame(self, frame_index: int = 0) -> np.ndarray:
        return self._image

    def close(self):
        return

    @staticmethod
    def check_is_valid_image(input_path: str | int) -> bool:
        if isinstance(input_path, str):
            return cv2.imread(input_path) is not None
        return False


class UploadedImageData(ImageAsVideoData):

    def __init__(self, raw_upload_buffer):
        raw_image = np.frombuffer(raw_upload_buffer, dtype=np.uint8)
        self._image = cv2.imdecode(raw_image, cv2.IMREAD_COLOR)
        assert self._image is not None, "Error reading upload buffer!"


class WebcamData(VideoData):
    """Class used to 'fake' a video file when using a webcam, which doesn't have a fixed frame count"""

    def __init__(self, video_path: int):
        self._vread: cv2.VideoCapture = create_video_capture(video_path)
        self._wh = (int(self._vread.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self._vread.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    def get_total_frames(self):
        return None

    def get_frame_index(self):
        return 0

    def set_frame_index(self, frame_index: int):
        # Do nothing, webcam cannot set a frame index
        return

    def read_frame(self, frame_index: int = 0) -> np.ndarray:

        ok_frame, frame = self._vread.read()
        assert ok_frame, "Error reading frame data from webcam!"

        return frame

    @staticmethod
    def check_is_webcam(input_path: str | int) -> bool:
        return isinstance(input_path, int)


@dataclass
class InputState:
    """Simple class used to hold global information about the input data source"""

    source_name = "unknown"
    data = None
    is_webcam = False
    is_video = False
    is_image = False
    image_wh = (0, 0)
    depth_wh = (0, 0)

    def read_input_path(self, input_path):
        """Helper used to properly interpret input data as image/video/webcam"""

        # Load data as if it is a video
        is_webcam = WebcamData.check_is_webcam(input_path)
        is_image = ImageAsVideoData.check_is_valid_image(input_path)
        is_video = not (is_webcam or is_image)
        if is_webcam:
            vreader = WebcamData(input_path)
        elif is_image:
            vreader = ImageAsVideoData(input_path)
        elif is_video:
            vreader = VideoData(input_path)
        else:
            raise TypeError("Unknown input data type!")

        # Store results
        self.vreader = vreader
        self.source_name = osp.basename(input_path) if not is_webcam else "webcam"
        self.is_webcam = is_webcam
        self.is_image = is_image
        self.is_video = is_video

        return self

    def update_sizing(self, dpt_image_preproc, dpt_model, device_config_dict, force_square_resolution=True):
        """Helper used to update internal record of image/depth sizing"""

        # Create example image to run through model
        image_wh = self.vreader.get_wh()
        ex_frame = np.random.randint(0, 255, [image_wh[1], image_wh[0], 3], dtype=np.uint8)
        ex_tensor = dpt_image_preproc.prepare_image_bgr(ex_frame, force_square_resolution).to(**device_config_dict)

        # Run model on example fraame to get output sizing
        ex_prediction = dpt_model.inference(ex_tensor)
        depth_wh = (ex_prediction.shape[2], ex_prediction.shape[1])

        # Store sizing
        self.image_wh = image_wh
        self.depth_wh = depth_wh

        return self


class MaskData(nn.Module):
    """
    Helper class used to manage loaded mask data or otherwise
    fall back to generating masks that target edges in depth
    predictions.
    This class is implemented as a pytorch 'model' to take
    advantage of device/dtype settings when generating edge
    detection images (i.e. we can use the gpu!).
    """

    def __init__(self, mask_path, mask_wh=None, blur_kernel_size=5, blur_weight=1):

        # Inherit from parent
        super().__init__()

        # Load mask image, if given
        self.path = mask_path
        ok_mask, mask_image = self._load_mask_image(mask_path, mask_wh)
        self.has_loaded_mask = ok_mask
        self.image = mask_image

        # Set up edge detection operations, for use if we don't have a mask
        self._blur = self._make_blur_filter(blur_kernel_size, blur_weight)
        self._derivative = self._make_derivative_filter()

        # Sanity check, make sure we're not tracking gradients
        self.requires_grad_(False)
        self.eval()

    def clear(self):
        self.path = None
        self.has_loaded_mask = False
        self.image = None

    def get_mask_uint8(self, depth_prediction):
        return self.image if self.has_loaded_mask else self.compute_edges_uint8(depth_prediction)

    def _load_mask_image(self, mask_path, mask_wh) -> tuple[bool, np.ndarray]:

        mask_image = None
        ok_path = mask_path is not None
        if not ok_path:
            return ok_path, mask_image

        # Try to read mask image
        mask_image = cv2.imread(mask_path)
        assert mask_image is not None, f"Unable to read mask image: {arg_mask_path}"

        # Convert to 1 channel (grayscale) mask
        mask_image = cv2.cvtColor(mask_image, cv2.COLOR_BGR2GRAY)
        mask_image = cv2.resize(mask_image, dsize=mask_wh)

        return ok_path, mask_image

    def compute_edges_uint8(self, depth_prediction):

        # Do x/y edge detection
        blur_pred = self._blur(depth_prediction)
        dxdy = self._derivative(blur_pred)

        # Combine dx & dy into a 'magnitude image'
        mag = torch.sqrt(torch.sum(torch.square(dxdy), dim=0))
        mag_uint8 = torch.bitwise_not(torch.round(255 * mag / mag.max()).byte())
        mag_uint8 = mag_uint8.squeeze().cpu().numpy()

        return mag_uint8

    @staticmethod
    def _make_derivative_filter():
        """Helper used to make a 2D derivative convolution operation (based on Sobel filter)"""

        # Build 2D derivative kernel
        sobel_kernel_dy = torch.tensor([[[[3, 10, 3], [0, 0, 0], [-3, -10, -3]]]], dtype=torch.float32)
        sobel_kernel_dx = sobel_kernel_dy.transpose(2, 3)
        sobel_kernel = torch.cat((sobel_kernel_dx, sobel_kernel_dy), dim=0)

        sobel = nn.Conv2d(1, 2, kernel_size=3, padding=1, padding_mode="reflect", bias=False)
        sobel.weight = nn.Parameter(sobel_kernel)
        sobel.requires_grad_(False)

        return sobel

    @staticmethod
    def _make_blur_filter(blur_kernel_size=5, blur_weight=1):
        """Helper used to make a gaussian blur convolution operation"""

        # Build 2D gaussian kernel
        ks_pad = blur_kernel_size // 2
        ksize = 1 + (2 * ks_pad)
        idx_1d = torch.linspace(-ks_pad, ks_pad, ksize, dtype=torch.float32)
        xy_idx = torch.stack(torch.meshgrid(idx_1d, idx_1d, indexing="ij"))
        gauss_scale = 0.01 / blur_weight
        gauss_kernel = torch.exp(-torch.sum(torch.square(xy_idx) * gauss_scale, dim=0))
        gauss_kernel = (gauss_kernel / gauss_kernel.max()).unsqueeze(0).unsqueeze(0)

        blur_conv = nn.Conv2d(1, 1, kernel_size=ksize, padding=ks_pad, padding_mode="reflect", bias=False)
        blur_conv.weight = nn.Parameter(gauss_kernel)
        blur_conv.requires_grad_(False)

        return blur_conv


# ---------------------------------------------------------------------------------------------------------------------
# %% Setup

# Make sure we can read the input
INDATA = InputState()
INDATA.read_input_path(input_path)

# Load model & image pre-processor
print("", f"Loading model weights ({osp.basename(model_path)})", sep="\n", flush=True)
model_config_dict, dpt_model, dpt_imgproc = make_dpt_from_state_dict(model_path, use_cache, use_optimizations)
if model_base_size is not None:
    dpt_imgproc.set_base_size(model_base_size)
device_config_dict = make_device_config(device_str, use_float32)
dpt_model.to(**device_config_dict)
dpt_model.eval()

# Set up globals for use in requests
MAX_UINT24 = (2**24) - 1
ENCODE_TYPE_RGB = arg_image_encoding if arg_image_encoding.startswith(".") else f".{arg_image_encoding}"
ENCODE_TYPE_DEPTH = arg_depth_encoding if arg_depth_encoding.startswith(".") else f".{arg_depth_encoding}"
DEPTH_IS_LOSSY = ENCODE_TYPE_DEPTH in {".jpg", "jpeg"}
BASE_FILES_PATH = osp.join(osp.dirname(__file__), "lib", "demo_helpers", "3dviewer")
VALID_FILES = set(os.listdir(BASE_FILES_PATH))
IS_METRIC_MODEL = model_config_dict.get("is_metric", False)

# Figure out image/depth sizing
INDATA.update_sizing(dpt_imgproc, dpt_model, device_config_dict, force_square_resolution)

# Load mask if given
MASKDATA = MaskData(arg_mask_path, INDATA.depth_wh).to(**device_config_dict)

# Some feedback for user
print(
    "",
    f"   Input: {INDATA.source_name}",
    f"Input WH: {INDATA.image_wh[0]} x {INDATA.image_wh[1]}",
    f"Depth WH: {INDATA.depth_wh[0]} x {INDATA.depth_wh[1]}",
    f"  Device: {device_config_dict.get('device', '...error...')}",
    f"   DType: {device_config_dict.get('dtype', '...error...')}",
    sep="\n",
)


# ---------------------------------------------------------------------------------------------------------------------
# %% *** Main server code ***


class RequestHandler(BaseHTTPRequestHandler):

    def do_GET(self):

        # Redirect to main page, if no path is given
        if self.path == "/":
            self.path = "/index.html"

        if self.path.startswith("/frame/"):

            # Update the current frame index (and jump video forward/backward if needed)
            frame_idx = int(self.path[7:])
            frame = INDATA.vreader.read_frame(frame_idx)

            # Run depth estimation
            frame_tensor = dpt_imgproc.prepare_image_bgr(frame, force_square_resolution)
            frame_tensor = frame_tensor.to(**device_config_dict)
            depth_prediction = dpt_model.inference(frame_tensor)
            if not IS_METRIC_MODEL:
                depth_prediction = dpt_imgproc.normalize_01(depth_prediction)
            depth_tensor_u24 = (torch.round(MAX_UINT24 * depth_prediction)).to(dtype=torch.int32)
            depth_u24 = depth_tensor_u24.squeeze().cpu().numpy()

            # Split depth bits into separate bytes to be stored in channels of RGB image
            # -> Have to do this because the browser does not directly support >8 bit images!
            # -> 24bit reconstruction is expected to take place on client end!
            # -> Only use upper-most bits if using lossy encoding (e.g. jpg), to reduce distortion
            depth_bgr = np.zeros((*depth_u24.shape[0:2], 4), dtype=np.uint8)
            depth_bgr[:, :, 2] = np.bitwise_and(np.right_shift(depth_u24, 16).astype(np.uint8), 255)
            if not DEPTH_IS_LOSSY:
                depth_bgr[:, :, 1] = np.bitwise_and(np.right_shift(depth_u24, 8).astype(np.uint8), 255)
                depth_bgr[:, :, 0] = np.bitwise_and(np.right_shift(depth_u24, 0).astype(np.uint8), 255)

            # Load mask into alpha channel of depth prediction
            depth_bgr[:, :, 3] = MASKDATA.get_mask_uint8(depth_prediction)

            # Convert images to (encoded) binary format for transfer
            ok_encode, encoded_rgb_img = cv2.imencode(ENCODE_TYPE_RGB, frame)
            if not ok_encode:
                raise ValueError(f"Error! Unable to encode frame as {ENCODE_TYPE_RGB}!")
            ok_encode, encoded_depth_img = cv2.imencode(ENCODE_TYPE_DEPTH, depth_bgr)
            self._set_image_headers(frame_idx, encoded_rgb_img.nbytes, encoded_depth_img.nbytes)

            try:
                self.wfile.write(encoded_rgb_img)
                self.wfile.write(encoded_depth_img)

            except BrokenPipeError:
                # Lost connection to client before we finished building our response
                # - Happens when frames are requested too quickly
                # - Nothing we can do (client can't hear us!), so just eat the error
                print("- connection closed before frame response...")
                pass
            pass

        elif self.path == "/get-source-info":

            info_dict = {
                "image_wh": INDATA.image_wh,
                "depth_wh": INDATA.depth_wh,
                "encode_type_rgb": ENCODE_TYPE_RGB,
                "encode_type_depth": ENCODE_TYPE_DEPTH,
                "total_frames": INDATA.vreader.get_total_frames(),
                "frame_index": INDATA.vreader.get_frame_index(),
                "is_static_image": INDATA.is_image,
                "is_webcam": INDATA.is_webcam,
                "is_video_file": INDATA.is_video,
                "is_metric_depth": IS_METRIC_MODEL,
                "source_name": INDATA.source_name,
            }

            self._set_simple_headers("text/json")
            self.wfile.write(json.dumps(info_dict).encode("utf-8"))

        elif self.path.startswith("/favicon"):
            # Draw the favicon every time it's requested, so we don't need a static file
            favicon_img = np.full((32, 32, 4), (0, 0, 0, 0), dtype=np.uint8)
            favicon_img = cv2.putText(favicon_img, "3D", (0, 23), 2, 0.75, (0, 0, 0, 255), 4)
            favicon_img = cv2.putText(favicon_img, "3D", (0, 23), 2, 0.75, (255, 255, 255, 255), 2)
            ok_encode, encoded_favicon = cv2.imencode(".png", favicon_img)
            self._set_simple_headers("image/x-icon")
            self.wfile.write(encoded_favicon)

        elif self.path.endswith(".html") or self.path.endswith(".js"):

            # Basic security check. Make sure we're not reaching for files outside of where we expect to load from
            file_path = self.path[1:]
            load_path = osp.join(BASE_FILES_PATH, file_path)
            if not (file_path in VALID_FILES and osp.exists(load_path)):
                self.send_error(404, f"Error! Unknown file path: {self.path}")
                return

            # Set content type based on file extention
            _, file_ext = osp.splitext(self.path)
            content_type = f"text/{file_ext[1:]}" if file_ext != ".js" else "text/javascript"
            self._set_simple_headers(content_type)

            # Respond with file contents
            with open(load_path, "r") as infile:
                self.wfile.write(infile.read().encode("utf-8"))
            pass

        else:
            self.send_error(404, f"Error! Unknown recognized request: {self.path}")
            return

        return

    def do_POST(self):

        if self.path.startswith("/upload"):

            # Read raw buffer data
            content_length = int(self.headers["Content-Length"])
            buffer_data = self.rfile.read(content_length)

            # Update data state
            INDATA.vreader = UploadedImageData(buffer_data)
            INDATA.is_image = True
            INDATA.is_video = False
            INDATA.is_webcam = False
            INDATA.source_name = self.headers.get("X-filename", "unknown")
            INDATA.update_sizing(dpt_imgproc, dpt_model, device_config_dict, force_square_resolution)

            # Reset masking so we don't re-use a loaded mask on a new image
            MASKDATA.clear()

            # Return ok response
            self._set_simple_headers("text/plain", 201)
            self.wfile.write("ok".encode("utf-8"))

        return

    def _set_simple_headers(self, content_type: str, response_code: int = 200):
        """Helper used to set up headers for 'simple' responses"""
        self.send_response(response_code)
        self.send_header("Content-type", content_type)
        self.end_headers()
        return

    def _set_image_headers(self, frame_index: int, rgb_size_bytes: int, depth_size_bytes: int):
        """Helper used to set up headers for image responses, which involve encoding binary sizing info"""
        self.send_response(200)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("X-rgb-size", str(rgb_size_bytes))
        self.send_header("X-depth-size", str(depth_size_bytes))
        self.send_header("X-frame-idx", frame_index)
        self.end_headers()
        return

    def log_request(self, code):
        # Override: prevents server from printing out a message for every request
        pass


# .....................................................................................................................


def get_server_ip() -> str:
    """
    Helper used to get the outwardly-visible IP address of the server
    Only meant to be used if server is accessible on network
    (i.e. server uses 0.0.0.0 host ip)
    """

    # Initialize guessed output (in case socket connection fails)
    ip_address = "127.0.0.1"

    # Weird OS-agnostic trick to get our own ip address
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.settimeout(0)
    try:
        s.connect(("10.255.255.255", 1))
        ip_address, _port = s.getsockname()
    finally:
        s.close()

    return ip_address


# ---------------------------------------------------------------------------------------------------------------------

# %% Run server

# Build server & provide feedback to user
server = HTTPServer((server_host, server_port), RequestHandler)
server_url = f"http://{server_host}:{server_port}"
if server_host == "0.0.0.0":
    server_url = f"http://{get_server_ip()}:{server_port}"
    print(
        "",
        "WARNING:",
        "This server does not have any serious security measures in place,",
        "nor does it operate efficiently. It is meant for a single user at a",
        "time, running on a local network for demo purposes only.",
        "Please do not use this in a production environment!",
        sep="\n",
    )
print("", "3D Viewer loaded in the browser:", server_url, sep="\n")

# Open page in browser, if needed
if launch_browser:
    webbrowser.open(server_url, new=0)

# Leave server running forever! Ctrl+c in terminal to end it
try:
    server.serve_forever()
except KeyboardInterrupt:
    pass
except Exception as err:
    raise err
finally:
    print("", "Closing...", sep="\n")
    server.server_close()
    INDATA.vreader.close()
