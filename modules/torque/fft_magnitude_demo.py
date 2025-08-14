import cv2, numpy as np, matplotlib.pyplot as plt
# import matplotlib
# matplotlib.use("Qt5Agg")  # or "Qt5Agg"
from mpl_toolkits.mplot3d import Axes3D

VIDEO = "/home/falthackel/Freelance/videos/raw/moire_test_data.mp4"
FRAME = 200                # any frame you like

# ---------- grab one frame ----------
cap = cv2.VideoCapture(VIDEO)
cap.set(cv2.CAP_PROP_POS_FRAMES, FRAME)
ok, bgr = cap.read();  cap.release()
gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)/255.

# ---------- FFT ----------
h, w  = gray.shape
opt_h = cv2.getOptimalDFTSize(h)
opt_w = cv2.getOptimalDFTSize(w)
padded = cv2.copyMakeBorder(gray, 0, opt_h-h, 0, opt_w-w,
                            cv2.BORDER_CONSTANT, value=0)
dft   = cv2.dft(padded, flags=cv2.DFT_COMPLEX_OUTPUT)
mag   = cv2.magnitude(dft[:,:,0], dft[:,:,1])
mag_shift = np.fft.fftshift(mag)          # move origin to centre

# ---------- plots ----------
plt.figure(figsize=(12,4))

plt.subplot(1,3,1)
plt.imshow(gray, cmap='gray')
plt.title('Original frame'); plt.axis('off')

plt.subplot(1,3,2)
plt.imshow(np.log(mag_shift+1), cmap='jet')
plt.title('log(|FFT|)'); plt.axis('off')

# 3-D surface of central region
plt.subplot(1,3,3, projection='3d')
Y, X = np.mgrid[-128:128, -128:128]
sub = mag_shift[opt_h//2-128:opt_h//2+128,
                opt_w//2-128:opt_w//2+128]
ax = plt.gca()
ax.plot_surface(X, Y, sub, cmap='jet')
ax.set_title('Central 256×256 |FFT| surface')

# plt.tight_layout()
# plt.show()

plt.tight_layout()
plt.savefig("fft_magnitude.png", dpi=300)
