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
import os
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

VIDEO_IN = "/home/falthackel/Freelance/videos/raw/moire_test_data.mp4"
VIDEO_OUT = "/home/falthackel/Freelance/videos/output/moire_output.mp4"


def optimal_size(n):
    return cv2.getOptimalDFTSize(n)

def grab_frame(index: int) -> np.ndarray:
    cap = cv2.VideoCapture(VIDEO_IN)
    cap.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, bgr = cap.read()
    cap.release()
    if not ok:
        raise IOError(f"Cannot read frame {index} from {VIDEO_IN}")
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return gray.astype(np.float32) / 255.0

def high_pass(img, sigma=5):
    """Remove low frequencies (DC) that hide the moiré peaks."""
    blur = cv2.GaussianBlur(img, (0, 0), sigma)
    return cv2.subtract(img, blur)

# ------------- put this in the same file, replace the old fft_polar ------------
def fft_polar(img,
              sigma=5,
              crop_ratio=0.65,
              r_min_pct=5,      # % of max_radius to discard (low freq)
              r_max_pct=95,     # % of max_radius to keep (high freq)
              sg_window=7):     # Savitzky-Golay smoothing window (0 = off)
    """
    Return angular profile, polar image, magnitude spectrum.
    Sub-pixel peak is later resolved by angle_from_profile.
    """

    # 1) high-pass and centre crop
    img_hp = high_pass(img, sigma)
    h, w = img_hp.shape
    margin = (1 - crop_ratio) / 2
    y0, y1 = int(margin * h), int((1 - margin) * h)
    x0, x1 = int(margin * w), int((1 - margin) * w)
    img_hp = img_hp[y0:y1, x0:x1]

    # 2) optimal DFT size
    opt_h = optimal_size(img_hp.shape[0])
    opt_w = optimal_size(img_hp.shape[1])
    padded = cv2.copyMakeBorder(img_hp,
                                0, opt_h - img_hp.shape[0],
                                0, opt_w - img_hp.shape[1],
                                cv2.BORDER_CONSTANT, 0)

    # 3) FFT magnitude
    dft = cv2.dft(padded, flags=cv2.DFT_COMPLEX_OUTPUT)
    mag = cv2.magnitude(dft[:, :, 0], dft[:, :, 1])
    mag = np.fft.fftshift(mag)
    mag = cv2.log(mag + 1)

    # 4) polar transform
    cy, cx = mag.shape[0] // 2, mag.shape[1] // 2
    max_radius = min(cy, cx)
    polar = cv2.warpPolar(mag,
                          (360, max_radius),
                          (cx, cy),
                          max_radius,
                          cv2.WARP_POLAR_LINEAR + cv2.INTER_LINEAR)

    # 5) radial mask
    r_min = max(1, int(max_radius * r_min_pct / 100))
    r_max = int(max_radius * r_max_pct / 100)
    mask = np.zeros_like(polar)
    mask[r_min:r_max, :] = 1
    polar *= mask

    # 6) radius-weighted angular profile
    radii = np.arange(max_radius).reshape(-1, 1)
    angular = np.sum(polar * radii, axis=0)  # weight ↑ with r

    # 7) smooth profile (optional)
    if sg_window >= 5 and len(angular) > sg_window:
        from scipy.signal import savgol_filter
        angular = savgol_filter(angular, sg_window, 2, mode='wrap')

    return angular, polar, mag


# ------------- replace angle_from_profile with this one ---------------
def angle_from_profile(profile):
    """
    Return sub-pixel angle [0-360) given a 360-bin profile.
    Uses parabolic interpolation around the integer peak.
    """
    idx = int(np.argmax(profile))
    N = len(profile)
    y0, y1, y2 = profile[(idx - 1) % N], profile[idx], profile[(idx + 1) % N]
    # parabolic vertex formula
    if y0 + y2 - 2 * y1 == 0:      # flat peak
        return float(idx)
    shift = 0.5 * (y0 - y2) / (y0 + y2 - 2 * y1)
    angle = (idx + shift) % N
    return angle if angle >= 0 else angle + N


if __name__ == "__main__":
    REF_IDX  = 0
    cap = cv2.VideoCapture(VIDEO_IN)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps          = cap.get(cv2.CAP_PROP_FPS)
    w            = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h            = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release() 

    # baseline
    prof_ref, _, _ = fft_polar(grab_frame(REF_IDX))
    theta_ref      = angle_from_profile(prof_ref)

    print("reference used in this run:", theta_ref)   # add once before the loop

    # containers
    angles_deg = []

    os.makedirs('/home/falthackel/Freelance/videos/output', exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(VIDEO_OUT, fourcc, fps, (w, h))

    for idx in tqdm(range(total_frames), desc="Processing frames"):
        img = grab_frame(idx)
        prof, _, _ = fft_polar(img)
        theta = angle_from_profile(prof)

        delta = (theta - theta_ref) % 360
        if delta > 180:
            delta -= 360
        angles_deg.append(delta)

        bgr = cv2.cvtColor((img*255).astype(np.uint8), cv2.COLOR_GRAY2BGR)  
        display_angle = delta        # relative change from frame 0
        
        # --- draw dot at (cx, cy) + radius * (cos, sin) ---
        radius = 30
        cx, cy = w // 2, h // 2
        rad = np.deg2rad(display_angle - 90)   # 0° = top
        dx  = int(radius * np.cos(rad))
        dy  = int(radius * np.sin(rad))
        cv2.circle(bgr, (cx + dx, cy + dy), 6, (0, 0, 255), -1)

        # --- overlay text ---
        text = f"Phase = {display_angle:+.2f}°"
        cv2.putText(bgr, text, (30, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 0, 255), 3)

        out.write(bgr)                                                        

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