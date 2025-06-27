import time
from pathlib import Path
from typing import List

import cv2
from tqdm import tqdm

from modules.processor import BaseProcessor


class VideoPipeline:
    """Main pipeline for processing videos with multiple processors"""

    def __init__(self, video_path: str, output_dir: str = "./output"):
        self.video_path = video_path
        self.output_dir = Path(output_dir)
        self.processors: List[BaseProcessor] = []

        self.cap = None
        self.writers = {}
        self.frame_count = 0
        self.fps = 0
        self.frame_size = None

    def add_processor(self, processor: BaseProcessor):
        """Add a processor to the pipeline"""
        self.processors.append(processor)

    def setup_video_io(self):
        """Setup video input and output"""
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            raise FileNotFoundError(f"Failed to open video: {self.video_path}")

        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)

        ret, first_frame = self.cap.read()
        if not ret:
            raise RuntimeError("Failed to read the first frame.")

        h, w, _ = first_frame.shape
        self.frame_size = (w, h)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        self.output_dir.mkdir(parents=True, exist_ok=True)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")

        for processor in self.processors:
            processor_dir = self.output_dir / processor.name
            processor_dir.mkdir(parents=True, exist_ok=True)

            output_specs = processor.get_output_specs()
            for output_name, spec in output_specs.items():
                if spec["type"] == "video":
                    writer_path = processor_dir / f"{output_name}.mp4"
                    writer = cv2.VideoWriter(
                        str(writer_path), fourcc, self.fps, self.frame_size
                    )
                    self.writers[f"{processor.name}_{output_name}"] = writer

    def run(self, show_display: bool = True):
        """Run the video processing pipeline"""
        self.setup_video_io()

        for processor in self.processors:
            processor.reset()

        ret = True

        with tqdm(total=self.frame_count, desc="Processing") as pbar:
            while ret:
                ret, frame = self.cap.read()
                if not ret:
                    break

                timestamp = time.time()

                for processor in self.processors:
                    results = processor.process_frame(frame, timestamp)
                    imshow_frames, video_frames = processor.visualize(frame, results)

                    # Handle video output
                    for output_name, output_frame in video_frames.items():
                        writer_key = f"{processor.name}_{output_name}"
                        if writer_key in self.writers:
                            if output_frame.shape[:2][::-1] != self.frame_size:
                                output_frame = cv2.resize(output_frame, self.frame_size)
                            self.writers[writer_key].write(output_frame)

                    # Handle display
                    if show_display:
                        for window_name, disp_frame in imshow_frames.items():
                            full_name = f"{processor.name} - {window_name}"
                            cv2.imshow(full_name, disp_frame)

                if show_display:
                    cv2.waitKey(1)

                pbar.update(1)

        # Cleanup
        self.cap.release()
        for writer in self.writers.values():
            writer.release()
        cv2.destroyAllWindows()
