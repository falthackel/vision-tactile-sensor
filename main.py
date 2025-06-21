import cv2

# Read video from file
cap = cv2.VideoCapture("data/test_video.mp4")
if not cap.isOpened():
    print("Error: Could not open video stream")
    exit()

# Get video properties
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30  # Fallback to 30 if 0
delay = int(1000 / fps)  # milliseconds

# Set up VideoWriter to save output
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter("output.mp4", fourcc, fps, (width, height))

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Display the processed frame
        cv2.imshow("Processed Frame", frame)
        cv2.waitKey(delay)

        # Write frame to output file
        out.write(frame)
finally:
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print("Video saved as 'output.mp4'")
