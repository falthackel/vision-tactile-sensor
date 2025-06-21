import os
import time

import cv2
import numpy as np
import torch
from tqdm import tqdm

from modules.shear.shear_estimator import ShearEstimator

METHOD = "weighted"
VIDEO_PATH = "./data/test_video.mp4"
OUTPUT_DIR = f"./output_videos_{METHOD}"
H_FIELD, W_FIELD = 13, 18


def setup_video_io():
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise FileNotFoundError(f"Failed to open video: {VIDEO_PATH}")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    ret, first_frame = cap.read()
    if not ret:
        raise RuntimeError("Failed to read the first frame.")

    h, w, _ = first_frame.shape
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    def make_writer(name):
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        return cv2.VideoWriter(
            os.path.join(OUTPUT_DIR, f"{name}.mp4"), fourcc, fps, (w, h)
        )

    writers = {
        "uv": make_writer("uv"),
        "dudv": make_writer("dudv"),
        "dudt_dvdt": make_writer("dudt_dvdt"),
        "solenoidal": make_writer("solenoidal"),
        "irrotational": make_writer("irrotational"),
        "div": make_writer("div"),
        "curl": make_writer("curl"),
    }

    return cap, writers, frame_count, fps, first_frame, h, w


def create_sampling_grid(h_video, w_video):
    y_coords = torch.linspace(0, h_video, H_FIELD)
    x_coords = torch.linspace(0, w_video, W_FIELD)
    return torch.meshgrid(y_coords, x_coords, indexing="ij")


def draw_vector_overlay(frame, u, v, grid_x, grid_y, scale=1.0):
    for i in range(H_FIELD):
        for j in range(W_FIELD):
            dx = float(u[i, j]) * scale
            dy = float(v[i, j]) * scale
            start_point = (int(grid_x[i, j]), int(grid_y[i, j]))
            end_point = (int(start_point[0] + dx), int(start_point[1] + dy))
            cv2.arrowedLine(
                frame,
                start_point,
                end_point,
                color=(0, 255, 0),
                thickness=3,
                tipLength=0.3,
            )
    return frame


def draw_scalar_overlay(frame, scalar):
    resized = cv2.resize(scalar.numpy(), (frame.shape[1], frame.shape[0]))
    normalized = cv2.normalize(resized, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    heatmap = cv2.applyColorMap(normalized, cv2.COLORMAP_JET)
    return cv2.addWeighted(frame, 0.6, heatmap, 0.4, 0)


def main():
    cap, writers, frame_count, fps, first_frame, h_video, w_video = setup_video_io()
    grid_y, grid_x = create_sampling_grid(h_video, w_video)

    base_tactile_image = torch.from_numpy(first_frame).permute(2, 0, 1).float()
    shgen = ShearEstimator(
        method=METHOD,
        channels=[
            "u",
            "v",
            "div",
            "curl",
            "sol_u",
            "sol_v",
            "irr_u",
            "irr_v",
            "dudt",
            "dvdt",
            "du",
            "dv",
        ],
        Farneback_params=(0.5, 3, 45, 3, 5, 1.2, 0),
    )
    shgen.reset_shear(base_tactile_image)

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ret = True

    overlay_map = {
        "uv": ("u", "v", 1.0),
        "dudv": ("du", "dv", 10.0),
        "dudt_dvdt": ("dudt", "dvdt", 10.0),
        "solenoidal": ("sol_u", "sol_v", 1.0),
        "irrotational": ("irr_u", "irr_v", 1.0),
    }

    with tqdm(total=frame_count, desc="Processing") as pbar:
        while ret:
            ret, frame_np = cap.read()
            if not ret:
                break

            tactile_image = torch.from_numpy(frame_np).permute(2, 0, 1).float()

            t = time.time()
            shgen.update_time(t)
            shgen.update_tactile_image(tactile_image)
            shgen.update_shear()
            shear_field_tensor = shgen.get_shear_field()

            display_rows = []

            # Vector overlays (top row)
            vector_row = []
            for name, (u_key, v_key, scale) in overlay_map.items():
                u = shear_field_tensor[shgen.channels.index(u_key)]
                v = shear_field_tensor[shgen.channels.index(v_key)]
                overlay = draw_vector_overlay(
                    frame_np.copy(), u, v, grid_x, grid_y, scale
                )
                resized_overlay = cv2.resize(overlay, (320, 240))
                vector_row.append(resized_overlay)
                writers[name].write(overlay)

            display_rows.append(np.hstack(vector_row))

            # Scalar overlays (bottom row)
            scalar_row = []
            for scalar in ["div", "curl"]:
                scalar_tensor = shear_field_tensor[shgen.channels.index(scalar)]
                overlay = draw_scalar_overlay(frame_np.copy(), scalar_tensor)
                resized_overlay = cv2.resize(overlay, (320, 240))
                scalar_row.append(resized_overlay)
                writers[scalar].write(overlay)

            # Pad scalar_row with black images to match vector_row length
            while len(scalar_row) < len(vector_row):
                scalar_row.append(np.zeros_like(scalar_row[0]))

            display_rows.append(np.hstack(scalar_row))

            # Final display frame (2 rows: vector + scalar)
            display_frame = np.vstack(display_rows)
            cv2.imshow("Shear Field Overlays", display_frame)

            cv2.waitKey(1)
            pbar.update(1)

    cap.release()
    for writer in writers.values():
        writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
