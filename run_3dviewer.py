#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# ---------------------------------------------------------------------------------------------------------------------
# %% Imports

import os
import os.path as osp
import json
import argparse
from http.server import HTTPServer, BaseHTTPRequestHandler
import webbrowser
import socket

import cv2
import numpy as np
import torch

from lib.make_dpt import make_dpt_from_state_dict

from lib.demo_helpers.history_keeper import HistoryKeeper
from lib.demo_helpers.loading import ask_for_path_if_missing, ask_for_model_path_if_missing
from lib.demo_helpers.misc import get_default_device_string, make_device_config


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
    "-d",
    "--device",
    default=default_device,
    type=str,
    help="Device to use when running model (ex: 'cpu', 'cuda', 'mps')",
)
parser.add_argument(
    "-f16",
    "--use_float16",
    default=False,
    action="store_true",
    help="Use 16-bit floating point model weights. This reduces prediction quality, but will run faster",
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
use_float32 = not args.use_float16
force_square_resolution = not args.use_aspect_ratio
model_base_size = args.base_size_px
server_host = args.host
server_port = args.port
use_webcam = args.use_webcam
launch_browser = args.launch
arg_image_encoding = str(args.encode_image).lower()
arg_depth_encoding = str(args.encode_depth).lower()

# Create history to re-use selected inputs
history = HistoryKeeper()
_, history_vidpath = history.read("video_path")
_, history_modelpath = history.read("model_path")

# Get pathing to resources, if not provided already
input_path = ask_for_path_if_missing(arg_input_path, "video", history_vidpath) if not use_webcam else 0
model_path = ask_for_model_path_if_missing(__file__, arg_model_path, history_modelpath)

# Store history for use on reload (but don't save video path when using webcam)
if use_webcam:
    history.store(model_path=model_path)
else:
    history.store(video_path=input_path, model_path=model_path)


# ---------------------------------------------------------------------------------------------------------------------
# %% Classes


class VideoData:
    """Class used to hold video reader/state data for handling video inputs"""

    def __init__(self, video_path: str):

        assert osp.exists(video_path), f"Error, video does not exist: {video_path}"
        self._vread: cv2.VideoCapture = cv2.VideoCapture(video_path)
        assert self._vread.isOpened(), f"Unable to open video: {video_path}"

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


class WebcamData(VideoData):
    """Class used to 'fake' a video file when using a webcam, which doesn't have a fixed frame count"""

    def __init__(self, video_path: int):
        self._vread: cv2.VideoCapture = cv2.VideoCapture(video_path)
        assert self._vread.isOpened(), f"Unable to open video: {video_path}"
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


# ---------------------------------------------------------------------------------------------------------------------
# %% Setup

# Load data as if it is a video
IS_WEBCAM_INPUT = WebcamData.check_is_webcam(input_path)
IS_IMAGE_INPUT = ImageAsVideoData.check_is_valid_image(input_path)
IS_VIDEO_INPUT = not (IS_WEBCAM_INPUT or IS_IMAGE_INPUT)
if IS_WEBCAM_INPUT:
    data = WebcamData(input_path)
elif IS_IMAGE_INPUT:
    data = ImageAsVideoData(input_path)
elif IS_VIDEO_INPUT:
    data = VideoData(input_path)
else:
    raise TypeError("Unknown input data type!")


# Load model & image pre-processor
print("", "Loading model weights...", "  @ {}".format(model_path), sep="\n", flush=True)
model_config_dict, dpt_model, dpt_imgproc = make_dpt_from_state_dict(model_path, enable_cache=True)
if model_base_size is not None:
    dpt_imgproc.set_base_size(model_base_size)
device_config_dict = make_device_config(device_str, use_float32)
dpt_model.to(**device_config_dict)
dpt_model.eval()

# Set up globals for use in requests
MAX_UINT24 = (2**24) - 1
SOURCE_NAME = osp.basename(input_path) if not IS_WEBCAM_INPUT else "webcam"
ENCODE_TYPE_RGB = arg_image_encoding if arg_image_encoding.startswith(".") else f".{arg_image_encoding}"
ENCODE_TYPE_DEPTH = arg_depth_encoding if arg_depth_encoding.startswith(".") else f".{arg_depth_encoding}"
DEPTH_IS_LOSSY = ENCODE_TYPE_DEPTH in {".jpg", "jpeg"}
BASE_FILES_PATH = osp.join(osp.dirname(__file__), "lib", "demo_helpers", "3dviewer")
VALID_FILES = set(os.listdir(BASE_FILES_PATH))
IS_METRIC_MODEL = model_config_dict.get("is_metric", False)

# Figure out image sizing
IMAGE_WH = data.get_wh()
ex_frame = np.random.randint(0, 255, [IMAGE_WH[1], IMAGE_WH[0], 3], dtype=np.uint8)
ex_tensor = dpt_imgproc.prepare_image_bgr(ex_frame, force_square_resolution).to(**device_config_dict)
ex_prediction = dpt_model.inference(ex_tensor)
DEPTH_WH = (ex_prediction.shape[2], ex_prediction.shape[1])


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
            frame = data.read_frame(frame_idx)

            # Run depth estimation
            frame_tensor = dpt_imgproc.prepare_image_bgr(frame, force_square_resolution)
            frame_tensor = frame_tensor.to(**device_config_dict)
            depth_prediction = dpt_model.inference(frame_tensor)
            if not IS_METRIC_MODEL:
                depth_prediction = dpt_imgproc.normalize_01(depth_prediction)
            depth_tensor_u24 = (torch.round(MAX_UINT24 * depth_prediction)).to(dtype=torch.uint32)
            depth_u24 = depth_tensor_u24.squeeze().cpu().numpy()

            # Split depth bits into separate bytes to be stored in red-green channels of RGB image
            # -> Have to do this because the browser does not directly support >8 bit images!
            # -> 24bit reconstruction is expected to take place on client end!
            # -> Only use upper-most bits if using lossy encoding (e.g. jpg), to reduce distortion
            depth_bgr = np.zeros((*depth_u24.shape[0:2], 3), dtype=np.uint8)
            depth_bgr[:, :, 2] = np.bitwise_and(np.right_shift(depth_u24, 16).astype(np.uint8), 255)
            if not DEPTH_IS_LOSSY:
                depth_bgr[:, :, 1] = np.bitwise_and(np.right_shift(depth_u24, 8).astype(np.uint8), 255)
                depth_bgr[:, :, 0] = np.bitwise_and(np.right_shift(depth_u24, 0).astype(np.uint8), 255)

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
                "image_wh": IMAGE_WH,
                "depth_wh": DEPTH_WH,
                "encode_type_rgb": ENCODE_TYPE_RGB,
                "encode_type_depth": ENCODE_TYPE_DEPTH,
                "total_frames": data.get_total_frames(),
                "frame_index": data.get_frame_index(),
                "is_static_image": IS_IMAGE_INPUT,
                "is_webcam": IS_WEBCAM_INPUT,
                "is_video_file": IS_VIDEO_INPUT,
                "is_metric_depth": IS_METRIC_MODEL,
                "source_name": SOURCE_NAME,
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

    def _set_simple_headers(self, content_type: str, response_code: int = 200):
        """Helper used to set up headers for 'simple' responses"""
        self.send_response(response_code)
        self.send_header("Content-type", content_type)
        self.end_headers()
        return

    def _set_image_headers(self, frame_index: int, rgb_image_size_bytes: int, depth_image_size_bytes: int):
        """Helper used to set up headers for image responses, which involve encoding binary sizing info"""
        self.send_response(200)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("X-rgb-size", str(rgb_image_size_bytes))
        self.send_header("X-depth-size", str(depth_image_size_bytes))
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
    data.close()
