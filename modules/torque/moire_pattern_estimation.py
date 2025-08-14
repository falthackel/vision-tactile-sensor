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

"""
Moire rotation measurement from a single video
FFT -> warpPolar -> arg-max
Dengan perbaikan stabilitas & akurasi
"""

import cv2
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.signal import medfilt, savgol_filter

VIDEO_IN = "/home/falthackel/Freelance/videos/raw/moire_test_data.mp4"
VIDEO_OUT = "/home/falthackel/Freelance/videos/output/moire_output.mp4"

# ---------- PARAMETER TUNING ----------
SIGMA_HP = 7                # Gaussian blur sigma untuk high-pass
R_MIN_PCT = 15              # persentase radius minimum
R_MAX_PCT = 85              # persentase radius maksimum
SG_WINDOW = 7               # Savitzky-Golay smoothing untuk angular profile
EMA_ALPHA = 0.1             # alpha untuk exponential moving average
MEDIAN_KERNEL = 5           # kernel size untuk median filter pasca proses
ROI_RATIO = 0.6             # rasio ROI (0.6 = ambil 60% tengah gambar)
RADIUS = 10                 # radius untuk visualisasi titik merah
# --------------------------------------

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

def high_pass(img, sigma=SIGMA_HP):
    """Hilangkan komponen frekuensi rendah."""
    blur = cv2.GaussianBlur(img, (0, 0), sigma)
    return cv2.subtract(img, blur)

def apply_window(img):
    """Terapkan windowing Hann untuk mengurangi ringing di FFT."""
    hann_y = np.hanning(img.shape[0])[:, None]
    hann_x = np.hanning(img.shape[1])
    return img * (hann_y * hann_x)

def fft_polar(img):
    """
    Return angular profile, polar image, magnitude spectrum.
    Dengan ROI + windowing + radial mask.
    """
    # 1) high-pass
    img_hp = high_pass(img, SIGMA_HP)

    # 2) ROI crop (ambil tengah)
    h, w = img_hp.shape
    margin = (1 - ROI_RATIO) / 2
    y0, y1 = int(margin * h), int((1 - margin) * h)
    x0, x1 = int(margin * w), int((1 - margin) * w)
    img_hp = img_hp[y0:y1, x0:x1]

    # 3) windowing
    img_hp = apply_window(img_hp)

    # 4) optimal DFT size + padding
    opt_h = optimal_size(img_hp.shape[0])
    opt_w = optimal_size(img_hp.shape[1])
    padded = cv2.copyMakeBorder(img_hp,
                                0, opt_h - img_hp.shape[0],
                                0, opt_w - img_hp.shape[1],
                                cv2.BORDER_CONSTANT, 0)

    # 5) FFT magnitude
    dft = cv2.dft(padded, flags=cv2.DFT_COMPLEX_OUTPUT)
    mag = cv2.magnitude(dft[:, :, 0], dft[:, :, 1])
    mag = np.fft.fftshift(mag)
    mag = cv2.log(mag + 1)

    # 6) polar transform
    cy, cx = mag.shape[0] // 2, mag.shape[1] // 2
    max_radius = min(cy, cx)
    polar = cv2.warpPolar(mag,
                          (360, max_radius),
                          (cx, cy),
                          max_radius,
                          cv2.WARP_POLAR_LINEAR + cv2.INTER_LINEAR)

    # 7) radial mask
    r_min = max(1, int(max_radius * R_MIN_PCT / 100))
    r_max = int(max_radius * R_MAX_PCT / 100)
    mask = np.zeros_like(polar)
    mask[r_min:r_max, :] = 1
    polar *= mask

    # 8) radius-weighted angular profile
    radii = np.arange(max_radius).reshape(-1, 1)
    angular = np.sum(polar * radii, axis=0)

    # 9) smooth profile (Savitzky-Golay)
    if SG_WINDOW >= 5 and len(angular) > SG_WINDOW:
        angular = savgol_filter(angular, SG_WINDOW, 2, mode='wrap')

    return angular, polar, mag

def angle_from_profile(profile):
    """Parabolic interpolation sub-pixel peak finding."""
    idx = int(np.argmax(profile))
    N = len(profile)
    y0, y1, y2 = profile[(idx - 1) % N], profile[idx], profile[(idx + 1) % N]
    if y0 + y2 - 2 * y1 == 0:
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

    # baseline reference for relative rotation
    prof_ref, _, _ = fft_polar(grab_frame(REF_IDX))
    theta_ref      = angle_from_profile(prof_ref)
    theta_abs0     = theta_ref  # baseline for absolute rotation zeroing
    print("reference used in this run:", theta_ref)

    # container hasil
    angles_rel_deg = []
    angles_abs_deg = []
    smoothed_angle_prev = None

    os.makedirs('/home/falthackel/Freelance/videos/output', exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(VIDEO_OUT, fourcc, fps, (w, h))

    for idx in tqdm(range(total_frames), desc="Processing frames"):
        img = grab_frame(idx)
        prof, _, _ = fft_polar(img)
        theta = angle_from_profile(prof)    # sudut absolut (0-360)

        # Zeroed absolute rotation
        abs_zeroed = (theta - theta_abs0 + 180) % 360 - 180

        # Relative rotation (zeroed + sign-flipped for CCW positive)
        delta = -((theta - theta_ref + 180) % 360 - 180)

        # Exponential moving average untuk smoothing real-time
        if smoothed_angle_prev is None:
            smoothed_angle = delta
        else:
            smoothed_angle = EMA_ALPHA * delta + (1 - EMA_ALPHA) * smoothed_angle_prev

        smoothed_angle_prev = smoothed_angle
        
        # Simpan kedua nilai
        angles_abs_deg.append(abs_zeroed)
        angles_rel_deg.append(smoothed_angle)

        # Visualisasi
        bgr = cv2.cvtColor((img*255).astype(np.uint8), cv2.COLOR_GRAY2BGR)  
        radius = RADIUS
        cx, cy = w // 2, h // 2
        rad = np.deg2rad(smoothed_angle - 90)
        dx  = int(radius * np.cos(rad))
        dy  = int(radius * np.sin(rad))
        cv2.circle(bgr, (cx + dx, cy + dy), 6, (0, 0, 255), -1)

        text = f"Rel = {smoothed_angle:+.2f}°, Abs = {theta:.2f}°"
        cv2.putText(bgr, text, (30, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 0, 255), 3)

        out.write(bgr) 

    out.release()

    # Median filter pasca proses
    angles_rel_deg = medfilt(angles_rel_deg, kernel_size=MEDIAN_KERNEL)
    angles_abs_deg = medfilt(angles_abs_deg, kernel_size=MEDIAN_KERNEL)

    # Simpan data
    df = pd.DataFrame({
        "frame"           : np.arange(total_frames),
        "time_s"          : np.arange(total_frames) / fps,
        "rotation_abs_deg": angles_abs_deg,
        "rotation_rel_deg": angles_rel_deg,
    })
    df.to_excel("moire_output.xlsx", index=False)
    print(f"Saved {len(df)} rows → moire_output.xlsx")
    print(f"Saved moire_output.mp4 in {VIDEO_OUT}")

    # # Smoothing dengan Savitzky-Golay
    # df["rotation_deg"] = savgol_filter(df["rotation_deg"], SG_WINDOW, 2, mode='wrap')

    # # Simpan data yang sudah dihaluskan
    # df.to_excel("moire_rotation_smoothed.xlsx", index=False)
    # print(f"Saved smoothed data → moire_rotation_smoothed.xlsx")

    # Print summary
    print(f"Total frames: {total_frames}, FPS: {fps}")
    print(f"Angle range: {df['rotation_rel_deg'].min():.2f}° to {df['rotation_rel_deg'].max():.2f}°")
    print(f"Time range: {df['time_s'].min():.2f}s to {df['time_s'].max():.2f}s")

    # Plot
    plt.figure(figsize=(10,4))
    plt.plot(df["time_s"], df["rotation_rel_deg"], label="Relative Rotation (°)")
    plt.xlabel("time  [s]")
    plt.ylabel("rotation  [°]")
    plt.title("Moire pattern rotation vs. time (smoothed)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("moire_output_plot.png", dpi=150)