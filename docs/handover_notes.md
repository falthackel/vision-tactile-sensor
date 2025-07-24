# Vision-Based Tactile Sensor – Handover Notes

This project estimates tactile signals (shear, normal force, and torque) using vision-based methods.

---

## 1. Shear Field Estimation

**Objective**: Estimate shear force (u, v) using dense optical flow from tactile images.

* **Reference project**: [https://sites.google.com/view/andreasipos/projects/gelslim-4-0](https://sites.google.com/view/andreasipos/projects/gelslim-4-0)
* **Original repo**: [https://github.com/MMintLab/gelslim\_shear](https://github.com/MMintLab/gelslim_shear)
* **Refined repo**: [https://github.com/shidqiet/gelslim-shear-sharing](https://github.com/shidqiet/gelslim-shear-sharing)

**Suggestions**:

* Understand how dense optical flow is converted into:

  * u, v displacement
  * du/dt, dv/dt (time derivatives)
  * etc
* Optical flow reference: [https://learnopencv.com/optical-flow-in-opencv/](https://learnopencv.com/optical-flow-in-opencv/)

---

## 2. Normal Force Estimation

**Objective**: Estimate normal force from tactile image.

* **Repo**: [https://github.com/shidqiet/normal-force-estimation](https://github.com/shidqiet/normal-force-estimation)

**Suggestions**:

* Read up on image preprocessing techniques:

  * Thresholding (global and adaptive)
  * Histogram equalization
  * CLAHE
  * Blob detection or `cv2.findContours` to identify contact regions.

---

## 3. Torque Estimation

**Objective**: Estimate torque from rotation patterns in the contact image.

* **Repo**: [https://github.com/shidqiet/torque-estimation-pattern-matching](https://github.com/shidqiet/torque-estimation-pattern-matching)

**Suggestions**:

* Current implementation uses synthetic data.
* Explore possibility of using real data and evaluate its performance.

---

## Final Integrated System

* **Final repo**: [https://github.com/shidqiet/vision-tactile-sensor](https://github.com/shidqiet/vision-tactile-sensor)
* All measurements (shear, normal, torque) are aimed to be processed in a single stream.