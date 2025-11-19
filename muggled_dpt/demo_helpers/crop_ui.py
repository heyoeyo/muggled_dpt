#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# ---------------------------------------------------------------------------------------------------------------------
# %% Imports

import numpy as np

from .toadui import (
    PrefixedTextBlock,
    TwoLineTextBlock,
    FixedARImage,
    EditBoxOverlay,
    ImmediateButton,
    VStack,
    HStack,
    DisplayWindow,
    MessageBar,
    VSeparator,
)


# ---------------------------------------------------------------------------------------------------------------------
# %% Functions


def run_crop_ui(
    image_bgr: np.ndarray,
    initial_crop_xy1xy2_norm: tuple[tuple[float, float], tuple[float, float]] | None = ((0.0, 0.0), (1.0, 1.0)),
    render_height: int = 600,
    window_title: str = "Crop Image - esc to close",
    minimum_crop_xy: tuple[int, int] = (5, 5),
) -> tuple[tuple[slice, slice], tuple[tuple[float, float], tuple[float, float]]]:
    """
    Helper used to launch a UI for cropping an image
    Returns:
        (y_crop_slice, x_crop_slice), crop_xy1xy2_norm

    - The slices are given in pixel units. To crop an image use:
        cropped_image = image[y_crop_slice, x_crop_slice]
    - The crop_xy1xy2_norm is a normalized (0 to 1) top-left/bottom-right box

    If the actual crop x/y values are needed, they can be accessed using:
        x1_px, x2_px = x_crop_slice.start, x_crop_slice.stop
        y1_px, y2_px = y_crop_slice.start, y_crop_slice.stop
    """

    # For convenience
    img_h, img_w = image_bgr.shape[0:2]
    is_tall_img = (img_h / img_w) > 1.35

    # Make ui elements
    TXTBlock = lambda title: TwoLineTextBlock(title, "?") if is_tall_img else PrefixedTextBlock(title, "?")
    orig_wh_txt = TXTBlock("Orig WH: ")
    crop_wh_txt = TXTBlock("Crop WH: ")
    xy1_txt = TXTBlock("XY1: ")
    xy2_txt = TXTBlock("XY2: ")
    img_elem = FixedARImage(image_bgr)
    box_olay = EditBoxOverlay(img_elem, color=(0, 255, 0))
    done_btn = ImmediateButton("Crop", color=(90, 205, 45), text_color=(255, 255, 255))
    msg_bar = MessageBar(
        "[Arrows] Adjust bounds", "[R-Click] Reset", color=70, use_equal_width=True, height=20, text_scale=0.35
    )

    # Build different UI for wide vs. tall images
    if is_tall_img:
        vsep1 = VSeparator(1, color=(40, 25, 30), is_flexible_h=True)
        vsep2 = VSeparator(8, color=(40, 25, 30), is_flexible_h=False)
        vsep3 = VSeparator(1, color=(40, 25, 30), is_flexible_h=True)
        crop_ui = VStack(
            HStack(
                box_olay,
                VStack(vsep1, orig_wh_txt, crop_wh_txt, vsep2, xy1_txt, xy2_txt, vsep3, done_btn),
                flex=(1, 0),
            ),
            msg_bar,
        )
    else:
        crop_ui = VStack(
            HStack(orig_wh_txt, crop_wh_txt),
            box_olay,
            HStack(xy1_txt, xy2_txt, done_btn, flex=(2, 2, 1)),
            msg_bar,
        )

    # Initialize cropping to use full image by default
    crop_xy1xy2_norm = tuple(((0.0, 0.0), (1.0, 1.0)))
    if initial_crop_xy1xy2_norm is not None:
        assert len(initial_crop_xy1xy2_norm) == 2, "Must provide crop coords as ((x1, y1), (x2,y2))"
        assert len(initial_crop_xy1xy2_norm[0]) == 2, "Must provide crop coords as ((x1, y1), (x2,y2))"
        box_olay.set_box(initial_crop_xy1xy2_norm)
        crop_xy1xy2_norm = crop_xy1xy2_norm

    # Set up window
    window = DisplayWindow(window_title, display_fps=60)
    window.enable_size_control(render_height)
    window.attach_mouse_callbacks(crop_ui)
    window.attach_keypress_callbacks(
        {
            "Adjust crop bounds": {
                "L_ARROW": lambda: box_olay.nudge(left=1),
                "R_ARROW": lambda: box_olay.nudge(right=1),
                "U_ARROW": lambda: box_olay.nudge(up=1),
                "D_ARROW": lambda: box_olay.nudge(down=1),
            },
            "Done": {"ENTER": done_btn.click},
        }
    ).report_keypress_descriptions(print_header="***** Cropping Controls *****")
    print("- Right click to reset cropping box", "- Shift+Drag to force start a new box", sep="\n", flush=True)

    try:

        # Set static image/sizing
        orig_wh_txt.set_text(f"({img_w}, {img_h})")
        img_elem.set_image(image_bgr)

        while True:

            # If we don't have a valid box set, then just crop to the full image
            is_crop_changed, is_valid_cropbox, crop_xy1xy2_norm = box_olay.read()
            if not is_valid_cropbox:
                crop_xy1xy2_norm = tuple(((0.0, 0.0), (1.0, 1.0)))

            # Update cropping coords whenever the crop changes
            if is_crop_changed:

                y_crop_slice, x_crop_slice = make_crop_slices_from_xy1xy2_norm(
                    image_bgr.shape, crop_xy1xy2_norm, minimum_crop_xy
                )

                # Crop the image!
                crop_image = image_bgr[y_crop_slice, x_crop_slice]
                crop_h_px, crop_w_px = crop_image.shape[0:2]

                # Update text indicators
                xy1_txt.set_text(f"({x_crop_slice.start}, {y_crop_slice.start})")
                xy2_txt.set_text(f"({x_crop_slice.stop}, {y_crop_slice.stop})")
                crop_wh_txt.set_text(f"({crop_w_px}, {crop_h_px})")

            # Update full display
            sizing = {"w": window.size} if is_tall_img else {"h": window.size}
            display_image = crop_ui.render(**sizing)  # w=window.size)
            req_break, keypress = window.show(display_image)
            if req_break:
                break

            # Finish when done is clicked
            is_done = done_btn.read()
            if is_done:
                break

    except KeyboardInterrupt:
        print("", "Crop cancelled!", sep="\n")
        crop_xy1xy2_norm = tuple(((0.0, 0.0), (1.0, 1.0)))

    except Exception as err:
        raise err

    finally:
        window.close()

    # Re-generate final slices (needed in case of errors)
    yx_crop_slices = make_crop_slices_from_xy1xy2_norm(image_bgr.shape, crop_xy1xy2_norm, minimum_crop_xy)
    return yx_crop_slices, crop_xy1xy2_norm


# .....................................................................................................................


def make_crop_slices_from_xy1xy2_norm(
    image_shape: tuple[int, int],
    crop_xy1xy2_norm: tuple[tuple[float, float], tuple[float, float]],
    minimum_crop_xy: tuple[int, int] = (5, 5),
) -> tuple[slice, slice]:
    """
    Helper used to convert normalized xy crop coordinates into 'slices'
    which can be used to easily crop an image
    Returns:
        y_crop_slice, x_crop_slice

    To crop an image index using slices:
        cropped_img = image[y_crop_slice, x_crop_slice]
    """

    # Get crop coordinates in pixel units with bounds clipping
    full_h, full_w = image_shape[0:2]
    max_wh = np.int32((full_w, full_h))
    crop_xy1xy2_px = np.int32(np.round(np.float32(crop_xy1xy2_norm) * np.float32(max_wh)))
    (cx1_px, cy1_px), (cx2_px, cy2_px) = np.clip(crop_xy1xy2_px, 0, max_wh)

    # Ignore overly small crops
    min_crop_x_px, min_crop_y_px = minimum_crop_xy
    if abs(cx2_px - cx1_px) < min_crop_x_px:
        cx1_px, cx2_px = 0, full_w
    if abs(cy2_px - cy1_px) < min_crop_y_px:
        cy1_px, cy2_px = 0, full_h

    # Bundle crop coords into slices for output
    x_crop_slice = slice(int(cx1_px), int(cx2_px))
    y_crop_slice = slice(int(cy1_px), int(cy2_px))
    return y_crop_slice, x_crop_slice
