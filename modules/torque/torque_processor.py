from pathlib import Path
from typing import Any, Dict

import cv2
import numpy as np

from modules.processor import BaseProcessor


class TorqueProcessor(BaseProcessor):
    """Processor for torque estimation using template matching"""

    def setup(self, **kwargs):
        self.templates_folder = Path(
            kwargs.get("templates_folder", "./modules/torque/templates")
        )
        self.threshold = kwargs.get("threshold", 1.0)
        self.method = kwargs.get("cv_method", cv2.TM_CCOEFF_NORMED)

        self.templates = self._load_templates()

    def _load_templates(self) -> Dict[str, Dict[str, Dict[str, Any]]]:
        """Load all template images organized by position folders"""
        templates = {}

        if not self.templates_folder.exists():
            print(f"Warning: Templates folder '{self.templates_folder}' not found")
            return templates

        # Get all subfolders (positions) under the template directory
        position_dirs = sorted(
            [d for d in self.templates_folder.iterdir() if d.is_dir()]
        )

        for position_path in position_dirs:
            position = position_path.name
            template_files = sorted(list(position_path.glob("*.png")))

            templates[position] = {}

            for template_file in template_files:
                # Extract angle from filename (format: bottomleft_template_{angle}.png)
                angle_str = template_file.stem.split("_")[-1]
                angle = int(angle_str)

                template_img = cv2.imread(str(template_file), cv2.IMREAD_GRAYSCALE)
                if template_img is None:
                    print(f"Skipping {template_file}: failed to load")
                    continue

                templates[position][angle] = template_img

        return templates

    def _resize_frame_to_template(
        self, frame: np.ndarray, template_size: int
    ) -> tuple[np.ndarray, float]:
        """Resize frame so its smaller dimension matches template size while keeping aspect ratio"""
        h, w = frame.shape[:2]
        smaller_dim = min(h, w)

        # Calculate scale factor to make smaller dimension equal to template size
        scale_factor = template_size / smaller_dim

        # Calculate new dimensions
        new_w = int(w * scale_factor)
        new_h = int(h * scale_factor)

        # Resize frame
        resized_frame = cv2.resize(
            frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR
        )

        return resized_frame, scale_factor

    def process_frame(self, frame: np.ndarray, timestamp: float) -> Dict[str, Any]:
        """Process frame for template matching"""
        if len(self.templates) == 0:
            return {"matches": []}

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Get template size (assuming square templates)
        first_position = next(iter(self.templates))
        first_angle = next(iter(self.templates[first_position]))
        template_size = self.templates[first_position][first_angle].shape[0]

        # Resize frame to match template size
        resized_gray_frame, scale_factor = self._resize_frame_to_template(
            gray_frame, template_size
        )

        position_best_match = {}

        # Process each position
        for position, angle_templates in self.templates.items():
            best_score = -1
            best_data = None

            # Test all angles for this position
            for angle, template in angle_templates.items():
                result = cv2.matchTemplate(resized_gray_frame, template, self.method)
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

                if max_val >= self.threshold:
                    if max_val > best_score:
                        # Calculate center on resized frame
                        center_x_resized = max_loc[0] + template.shape[1] // 2
                        center_y_resized = max_loc[1] + template.shape[0] // 2

                        # Scale coordinates back to original frame size
                        center_x = int(center_x_resized / scale_factor)
                        center_y = int(center_y_resized / scale_factor)
                        top_left_x = int(max_loc[0] / scale_factor)
                        top_left_y = int(max_loc[1] / scale_factor)
                        size_w = int(template.shape[1] / scale_factor)
                        size_h = int(template.shape[0] / scale_factor)

                        best_score = max_val
                        best_data = {
                            "position": position,
                            "angle": angle,
                            "confidence": max_val,
                            "center": (center_x, center_y),
                            "top_left": (top_left_x, top_left_y),
                            "size": (size_w, size_h),
                        }

            if best_data:
                position_best_match[position] = best_data

        # Convert to list format for consistency with original processor
        matches = list(position_best_match.values())

        # Sort matches by confidence
        matches.sort(key=lambda x: x["confidence"], reverse=True)

        return {"matches": matches, "position_matches": position_best_match}

    def visualize(self, frame: np.ndarray, results: Dict[str, Any]) -> np.ndarray:
        """Create template matching visualization"""
        vis_frame = frame.copy()

        # Draw only the best match for each detected position
        for match in results["matches"]:
            center_x, center_y = match["center"]
            position = match["position"]
            angle = match["angle"]

            # Draw circle at match center (larger circle like in original)
            cv2.circle(vis_frame, (center_x, center_y), 128, (0, 0, 255), 10)

            # Add text with position and angle
            text = f"{position}_{angle}"
            cv2.putText(
                vis_frame,
                text,
                (center_x, center_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                2,
                (0, 0, 255),
                3,
            )

        return {"torque_output": vis_frame}, {"template_matches": vis_frame}

    def get_output_specs(self) -> Dict[str, Dict[str, Any]]:
        """Return output file specifications"""
        return {
            "template_matches": {
                "type": "video",
                "description": "Template matching visualization with position and angle detection",
            }
        }
