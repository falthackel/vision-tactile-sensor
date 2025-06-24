from typing import Any, Dict, Tuple

import cv2
import numpy as np
import torch

from modules.processor import BaseProcessor


class ShearProcessor(BaseProcessor):
    """Processor for shear field estimation"""

    def setup(self, **kwargs):
        from modules.shear.shear_estimator import ShearEstimator

        self.method = kwargs.get("method", "weighted")
        self.h_field = kwargs.get("h_field", 13)
        self.w_field = kwargs.get("w_field", 18)

        self.shgen = ShearEstimator(
            method=self.method,
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

        self.overlay_map = {
            "uv": ("u", "v", 1.0),
            "dudv": ("du", "dv", 10.0),
            "dudt_dvdt": ("dudt", "dvdt", 10.0),
            "solenoidal": ("sol_u", "sol_v", 1.0),
            "irrotational": ("irr_u", "irr_v", 1.0),
        }

        self.grid_y = None
        self.grid_x = None
        self.initialized = False

    def _initialize_grid(self, h_video: int, w_video: int):
        """Initialize sampling grid based on video dimensions"""
        y_coords = torch.linspace(0, h_video, self.h_field)
        x_coords = torch.linspace(0, w_video, self.w_field)
        self.grid_y, self.grid_x = torch.meshgrid(y_coords, x_coords, indexing="ij")

    def process_frame(self, frame: np.ndarray, timestamp: float) -> Dict[str, Any]:
        """Process frame for shear field estimation"""
        if not self.initialized:
            h, w, _ = frame.shape
            self._initialize_grid(h, w)
            base_tactile_image = torch.from_numpy(frame).permute(2, 0, 1).float()
            self.shgen.reset_shear(base_tactile_image)
            self.initialized = True

        tactile_image = torch.from_numpy(frame).permute(2, 0, 1).float()

        self.shgen.update_time(timestamp)
        self.shgen.update_tactile_image(tactile_image)
        self.shgen.update_shear()
        shear_field_tensor = self.shgen.get_shear_field()

        return {"shear_field": shear_field_tensor}

    def visualize(
        self, frame: np.ndarray, results: Dict[str, Any]
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """Create shear field visualization outputs"""

        from modules.shear.shear_visualizer import (
            draw_scalar_overlay,
            draw_vector_overlay,
        )

        shear_field = results["shear_field"]
        imshow_dict = {}
        video_dict = {}

        # Vector overlays (top row)
        vector_row = []
        for name, (u_key, v_key, scale) in self.overlay_map.items():
            u = shear_field[self.shgen.channels.index(u_key)]
            v = shear_field[self.shgen.channels.index(v_key)]
            overlay = draw_vector_overlay(
                frame.copy(),
                u,
                v,
                self.grid_x,
                self.grid_y,
                self.h_field,
                self.w_field,
                scale,
            )
            resized = cv2.resize(overlay, (320, 240))
            video_dict[name] = resized
            vector_row.append(resized)

        # Scalar overlays (bottom row)
        scalar_row = []
        for scalar in ["div", "curl"]:
            scalar_tensor = shear_field[self.shgen.channels.index(scalar)]
            overlay = draw_scalar_overlay(frame.copy(), scalar_tensor)
            resized = cv2.resize(overlay, (320, 240))
            video_dict[scalar] = resized
            scalar_row.append(resized)

        while len(scalar_row) < len(vector_row):
            scalar_row.append(np.zeros_like(scalar_row[0]))

        combined = np.vstack([np.hstack(vector_row), np.hstack(scalar_row)])
        imshow_dict["shear_output"] = combined

        return imshow_dict, video_dict

    def get_output_specs(self) -> Dict[str, Dict[str, Any]]:
        """Return output file specifications"""
        specs = {}

        for name in self.overlay_map.keys():
            specs[name] = {
                "type": "video",
                "description": f"{name} vector field overlay",
            }

        for scalar in ["div", "curl"]:
            specs[scalar] = {
                "type": "video",
                "description": f"{scalar} scalar field overlay",
            }

        return specs

    def reset(self):
        """Reset processor state"""
        self.initialized = False
