"""
Moire rotation measurement from a single video
FFT -> warpPolar -> arg-max

Input images
     │            ┌------------┐
     ├------------►│  FFT(2D)   │───┐
     │            └------------┘   │
     │                             ▼
     │            ┌------------┐  Angular
     └------------►│warpPolar   │  profile
                   └------------┘
                          │
                          ▼
                   Peak in 1-D spectrum
                          │
                          ▼
                   θ = arg-max
"""

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

VIDEO = "/home/falthackel/Freelance/videos/raw/moire_test_data.mp4"


def optimal_size(n):
    return cv2.getOptimalDFTSize(n)

def grab_frame(index: int) -> np.ndarray:
    cap = cv2.VideoCapture(VIDEO)
    cap.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, bgr = cap.read()
    cap.release()
    if not ok:
        raise IOError(f"Cannot read frame {index} from {VIDEO}")
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return gray.astype(np.float32) / 255.0

def high_pass(img, sigma=5):
    """Remove low frequencies (DC) that hide the moiré peaks."""
    blur = cv2.GaussianBlur(img, (0, 0), sigma)
    return cv2.subtract(img, blur)

def fft_polar(img):
    # 1) pre-filter & crop
    img = high_pass(img, sigma=5)
    h, w = img.shape
    # crop central 65 %
    y0, y1 = int(0.175 * h), int(0.825 * h)
    x0, x1 = int(0.175 * w), int(0.825 * w)
    img = img[y0:y1, x0:x1]

    # 2) pad to optimal FFT size
    padded = cv2.copyMakeBorder(
        img,
        0, optimal_size(img.shape[0]) - img.shape[0],
        0, optimal_size(img.shape[1]) - img.shape[1],
        cv2.BORDER_CONSTANT,
        value=0,
    )

    # 3) FFT magnitude
    dft = cv2.dft(padded, flags=cv2.DFT_COMPLEX_OUTPUT)
    mag = cv2.magnitude(dft[:, :, 0], dft[:, :, 1])
    mag = np.fft.fftshift(mag)
    mag = cv2.log(mag + 1)

    # 4) polar transform
    cy, cx = mag.shape[0] // 2, mag.shape[1] // 2
    max_radius = min(cy, cx)
    polar = cv2.warpPolar(
        mag,
        (360, max_radius),
        (cx, cy),
        max_radius,
        cv2.WARP_POLAR_LINEAR,
    )

    # 5) angular profile
    angular = np.sum(polar, axis=0)
    return angular, polar, mag

def angle_from_profile(profile):
    return np.argmax(profile)  # degrees


if __name__ == "__main__":
    REF_IDX  = 0
    cap = cv2.VideoCapture(VIDEO)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps          = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    # baseline
    prof_ref, _, _ = fft_polar(grab_frame(REF_IDX))
    theta_ref      = angle_from_profile(prof_ref)

    # containers
    angles_deg = []

    for idx in tqdm(range(total_frames), desc="Processing frames"):
        prof, _, _ = fft_polar(grab_frame(idx))
        theta      = angle_from_profile(prof)

        delta = (theta - theta_ref) % 360
        if delta > 180:
            delta -= 360
        angles_deg.append(delta)

    # build dataframe
    df = pd.DataFrame({
        "frame"   : np.arange(total_frames),
        "time_s"  : np.arange(total_frames) / fps,
        "rotation_deg": angles_deg,
    })
    df.to_excel("moire_rotation.xlsx", index=False)
    print(f"Saved {len(df)} rows → moire_rotation.xlsx")

    # quick plot
    plt.figure(figsize=(10,4))
    plt.plot(df["time_s"], df["rotation_deg"])
    plt.xlabel("time  [s]")
    plt.ylabel("rotation  [°]")
    plt.title("Moire pattern rotation vs. time")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("rotation_plot.png", dpi=150)
    plt.show()