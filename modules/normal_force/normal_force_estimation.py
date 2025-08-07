import cv2
import matplotlib.pyplot as plt
import pandas as pd

THRESHOLD_VALUE = 50
MIN_AREA = 1000


def is_square_like(w, h):
    aspect_ratio = float(w) / h if h != 0 else 0
    return 0.75 < aspect_ratio < 4


def find_closest_centroid(cx, cy, reference_blobs):
    min_dist = float("inf")
    min_index = -1
    for i, (rcx, rcy, *_rest) in enumerate(reference_blobs):
        dist = (cx - rcx) ** 2 + (cy - rcy) ** 2
        if dist < min_dist:
            min_dist = dist
            min_index = i
    return min_index


def main():
    cap = cv2.VideoCapture("/home/falthackel/Freelance/videos/raw/VID-20250803-WA0011.mp4")
    if not cap.isOpened():
        print("Error: Could not open video source")
        return

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Cropping slice
    slice_width = (125, 950)
    slice_height = (45, 675)

    # Video writers
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_thresh = cv2.VideoWriter(
        "thresholded_output.mp4",
        fourcc,
        fps,
        # (slice_width[1] - slice_width[0], slice_height[1] - slice_height[0]),
        (width, height),
    )
    out_annotated = cv2.VideoWriter(
        "annotated_original_output.mp4", fourcc, fps, (width, height)
    )

    blob_history = {}
    initialized = False
    frame_idx = 0
    data_rows = []

    while True:
        ret, full_frame = cap.read()
        if not ret:
            break

        # ----------------------------------------------------------
        # 1.  PROCESS THE WHOLE FRAME
        # ----------------------------------------------------------

        cropped_frame = full_frame[
            slice_height[0] : slice_height[1], slice_width[0] : slice_width[1]
        ]
        # gray = cv2.cvtColor(full_frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, THRESHOLD_VALUE, 255, cv2.THRESH_BINARY)
        inverted = cv2.bitwise_not(thresh)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        # cleaned = cv2.dilate(inverted, kernel, iterations = 3)
        cleaned = cv2.morphologyEx(inverted, cv2.MORPH_CLOSE, kernel, iterations=3)
        
        # # 1. Kill lone specks (opening)
        # kernel_open  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        # opened = cv2.morphologyEx(inverted, cv2.MORPH_OPEN, kernel_open, iterations=1)

        # # 2. Glue fragments (closing)
        # kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))   # wide enough to bridge gaps
        # closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel_close, iterations=1)

        # # 3. Optional final dilation to smooth edges
        # kernel_final = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        # cleaned = cv2.dilate(closed, kernel_final, iterations=1)

        # For thresholded output video
        thresh_bgr = cv2.cvtColor(cleaned, cv2.COLOR_GRAY2BGR)

        contours, _ = cv2.findContours(
            cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # ----------------------------------------------------------
        # 2.  BUILD CURRENT_BLOBS ON THE FULL FRAME
        # ----------------------------------------------------------

        current_blobs = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h
            if area < MIN_AREA or not is_square_like(w, h):
                continue
            cx, cy = x + w // 2, y + h // 2     # centroid in full-frame coords
            current_blobs.append((cx, cy, w, h))

        # ----------------------------------------------------------
        # 3.  INITIALISE ON THE *FIRST* FRAME WITH NON-ZERO BLOBS
        # ----------------------------------------------------------

        if not initialized and len(current_blobs) > 0:
            initial_blobs = current_blobs.copy()
            for i in range(len(initial_blobs)):
                blob_history[i] = {"width": [], "height": []}
            initialized = True
            
        # ----------------------------------------------------------
        # 4.  DRAW ON FULL FRAME
        # ----------------------------------------------------------

        timestamp_ms = frame_idx * (1000 // fps)
        row = {"timestamp_ms": timestamp_ms}

        for cx, cy, w, h in current_blobs:
            idx = find_closest_centroid(cx, cy, initial_blobs)
            if idx == -1:
                continue
            init_w, init_h = initial_blobs[idx][2], initial_blobs[idx][3]
            dw, dh = w - init_w, h - init_h
            dw = max(0, dw)
            dh = max(0, dh)

            # Store data for every blob
            blob_history[idx]["width"].append(dw)
            blob_history[idx]["height"].append(dh)
            row[f"blob_{idx}_dW"] = dw
            row[f"blob_{idx}_dH"] = dh

            # Red rectangle + label on the full frame
            cv2.rectangle(
                thresh_bgr,
                (cx - w // 2, cy - h // 2),
                (cx + w // 2, cy + h // 2),
                (255, 0, 0),
                2,
            )
            cv2.putText(
                thresh_bgr,
                f"dW:{dw} dH:{dh}",
                (cx - w // 2, cy - h // 2 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (0, 255, 0),
                1,
            )
            # 4-A  Annotated original (RED rectangle)
            cv2.rectangle(
                cropped_frame,
                (cx - w // 2, cy - h // 2),
                (cx + w // 2, cy + h // 2),
                (0, 0, 255), 2
            )
            cv2.putText(
                cropped_frame,
                f"dW:{dw} dH:{dh}",
                (cx - w // 2, cy - h // 2 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                1,
            )

            # ----------------------------------------------------------
            # 5.  THRESHOLD VIDEO (optional) – same as before
            # ----------------------------------------------------------
            thresh_bgr = cv2.cvtColor(cleaned, cv2.COLOR_GRAY2BGR)
            for cx, cy, w, h in current_blobs:
                idx = find_closest_centroid(cx, cy, initial_blobs)
                if idx == -1:
                    continue

                # Re-calculate dW / dH exactly like above
                init_w, init_h = initial_blobs[idx][2], initial_blobs[idx][3]
                dw, dh = max(0, w - init_w), max(0, h - init_h)

                # Draw rectangle and text on the thresholded video
                cv2.rectangle(
                    thresh_bgr,
                    (cx - w // 2, cy - h // 2),
                    (cx + w // 2, cy + h // 2),
                    (0, 0, 255), 1
                )
                cv2.putText(
                    thresh_bgr, f"dW:{dw} dH:{dh}",
                    (cx - w // 2, cy - h // 2 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1
                )

        data_rows.append(row)

        out_thresh.write(thresh_bgr)
        out_annotated.write(full_frame)

        cv2.imshow("Thresholded Blobs", thresh_bgr)
        cv2.imshow("Annotated Original", full_frame)
        cv2.waitKey(1)
        frame_idx += 1

    cap.release()
    out_thresh.release()
    out_annotated.release()
    cv2.destroyAllWindows()

    # Save data
    df = pd.DataFrame(data_rows)
    df.to_excel("blob_displacement.xlsx", index=False)

    # Plotting
    fig, ax = plt.subplots(figsize=(12, 6))
    for idx, data in blob_history.items():
        ax.plot(data["width"], label=f"Blob {idx} dW")
        ax.plot(data["height"], label=f"Blob {idx} dH", linestyle="--")

    ax.set_title("Blob Dimension Displacement Over Time")
    ax.set_xlabel("Frame")
    ax.set_ylabel("Δpixels")
    ax.legend()
    plt.tight_layout()
    plt.savefig("blob_displacement_plot.png")


if __name__ == "__main__":
    main()