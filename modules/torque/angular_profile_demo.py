import cv2, numpy as np, matplotlib.pyplot as plt
from fft_magnitude_demo import gray   # reuse the frame we already loaded

# ---------- reuse fft_polar ----------
def fft_polar(img):
    img = cv2.GaussianBlur(img, (0,0), 5)
    img = cv2.subtract(img, cv2.GaussianBlur(img, (0,0), 5))
    h, w = img.shape
    y0,y1 = int(.175*h), int(.825*h); x0,x1 = int(.175*w), int(.825*w)
    img = img[y0:y1, x0:x1]
    opt_h = cv2.getOptimalDFTSize(img.shape[0])
    opt_w = cv2.getOptimalDFTSize(img.shape[1])
    padded = cv2.copyMakeBorder(img,0,opt_h-img.shape[0],0,opt_w-img.shape[1],
                                cv2.BORDER_CONSTANT,0)
    dft = cv2.dft(padded, flags=cv2.DFT_COMPLEX_OUTPUT)
    mag = cv2.magnitude(dft[:,:,0], dft[:,:,1])
    mag = np.fft.fftshift(mag)
    mag = cv2.log(mag+1)

    cy,cx = mag.shape[0]//2, mag.shape[1]//2
    max_r = min(cy,cx)
    polar = cv2.warpPolar(mag,(360,max_r),(cx,cy),max_r,
                          cv2.WARP_POLAR_LINEAR)
    angular = np.sum(polar, axis=0)
    return angular, polar, mag

ang, pol, mag = fft_polar(gray)

# ---------- visualise ----------
fig, ax = plt.subplots(1,3, figsize=(15,4))
ax[0].imshow(mag, cmap='jet');  ax[0].set_title('log|FFT| (Cartesian)')
ax[1].imshow(pol, cmap='jet', aspect='auto');  ax[1].set_title('Polar image (rows=r, cols=θ)')
ax[2].plot(ang);  ax[2].set_title('Angular profile');  ax[2].set_xlabel('θ [°]')
plt.tight_layout();  plt.savefig("angular_profile.png", dpi=300)

print("Dominant angle =", np.argmax(ang), "°")