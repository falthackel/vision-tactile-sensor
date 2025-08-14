import cv2, numpy as np, pandas as pd
from math import cos, sin, pi
import os

VIDEO_IN = "/home/falthackel/Freelance/videos/raw/moire_test_data.mp4"
VIDEO_OUT = "/home/falthackel/Freelance/videos/output/moire_output.mp4"
DATA      = "moire_rotation.xlsx"

# Read input video properties
cap = cv2.VideoCapture(VIDEO_IN)
FPS = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(f"Input video: {width}x{height} @ {FPS}fps")

# Load rotation data
df = pd.read_excel(DATA)
angles = df['rotation_deg'].values

# Try different codec options in order of preference
codec_options = [
    ('mp4v', '.mp4'),  # MP4V codec with MP4 container
    ('XVID', '.avi'),  # XVID codec with AVI container  
    ('MJPG', '.avi'),  # MJPG codec with AVI container
]

out = None
final_output_path = VIDEO_OUT

for fourcc_str, extension in codec_options:
    try:
        fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
        
        # If not the first option, change extension
        if extension != '.mp4':
            base_path = VIDEO_OUT.rsplit('.', 1)[0]
            test_output_path = base_path + extension
        else:
            test_output_path = VIDEO_OUT
            
        out = cv2.VideoWriter(test_output_path, fourcc, FPS, (width, height))
        
        # Test if writer was created successfully
        if out.isOpened():
            print(f"Successfully initialized VideoWriter with {fourcc_str} codec")
            final_output_path = test_output_path
            break
        else:
            out.release()
            out = None
            
    except Exception as e:
        print(f"Failed to initialize with {fourcc_str}: {e}")
        if out:
            out.release()
            out = None

if out is None:
    print("Error: Could not initialize VideoWriter with any codec")
    cap.release()
    exit(1)

# Reset video capture
cap.release()
cap = cv2.VideoCapture(VIDEO_IN)

radius = 0   # pixels from centre

print(f"Processing {len(angles)} frames...")

for idx, angle in enumerate(angles):
    ok, frame = cap.read()
    if not ok: 
        print(f"Failed to read frame {idx}")
        break
        
    h, w = frame.shape[:2]
    cx, cy = w//2, h//2
    
    # Calculate dot position
    dx = int(radius * cos((angle-90)*pi/180))
    dy = int(radius * sin((angle-90)*pi/180))
    
    # Draw red dot
    cv2.circle(frame, (cx+dx, cy+dy), 6, (0,0,255), -1)
    
    # Draw angle text
    cv2.putText(frame, f"{angle:+.1f}°", (cx-50, cy-60),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,255), 2)
    
    # Write frame
    out.write(frame)
    
    # Progress indicator
    if (idx + 1) % 30 == 0:  # Every 30 frames
        print(f"Processed {idx + 1}/{len(angles)} frames")

# Clean up
cap.release()
out.release()

# Verify output file exists and has reasonable size
if os.path.exists(final_output_path):
    file_size = os.path.getsize(final_output_path)
    print(f"Successfully saved: {final_output_path}")
    print(f"Output file size: {file_size / (1024*1024):.2f} MB")
    
    # Quick verification - try to open the output file
    test_cap = cv2.VideoCapture(final_output_path)
    frame_count = int(test_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    test_cap.release()
    print(f"Output video frame count: {frame_count}")
else:
    print(f"Error: Output file was not created at {final_output_path}")