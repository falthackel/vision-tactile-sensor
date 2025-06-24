from abc import ABC, abstractmethod
from typing import Any, Dict

import numpy as np


class BaseProcessor(ABC):
    """Abstract base class for video processors"""

    def __init__(self, name: str, **kwargs):
        self.name = name
        self.kwargs = kwargs
        self.setup(**kwargs)

    @abstractmethod
    def setup(self, **kwargs):
        """Initialize processor-specific setup"""
        pass

    @abstractmethod
    def process_frame(self, frame: np.ndarray, timestamp: float) -> Dict[str, Any]:
        """Process a single frame and return results"""
        pass

    @abstractmethod
    def visualize(self, frame: np.ndarray, results: Dict[str, Any]) -> np.ndarray:
        """Create visualization overlay on frame"""
        pass

    @abstractmethod
    def get_output_specs(self) -> Dict[str, Dict[str, Any]]:
        """Return specifications for output files this processor generates"""
        pass

    def reset(self):
        """Reset processor state if needed"""
        pass
