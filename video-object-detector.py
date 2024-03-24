import cv2
from transformers import DetrImageProcessor, DetrForObjectDetection
import threading
import time

# Load DETR model and image processor
processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-101")
model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-101")

# Global variables for storing the latest frame and processed result
latest_frame = None
latest_result = None

# Function for processing frames asynchronously
def process_frames():
    global latest_frame, latest_result

    while True:
        if latest_frame is not None:
            # Process the latest frame
            frame = latest_frame.copy()  # Make a copy to avoid modifying the original frame
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            inputs = processor(images=rgb_frame, return_tensors="pt")
            outputs = model(**inputs)
            results = processor.post_process_object_detection(outputs, threshold=0.9)[0]
            latest_result = results

        # Sleep for a short duration to avoid busy waiting
        time.sleep(0.01)

# Start the frame processing thread
thread = threading.Thread(target=process_frames)
thread.daemon = True  # Daemonize the thread so it automatically exits when the main program exits
thread.start()

# Open webcam
cap = cv2.VideoCapture(0)

# Set width and height
width = 800
height = 800
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Store the latest frame
    latest_frame = frame

    # Display frame with bounding boxes (if available)
    if latest_result is not None:
        for score, label, box in zip(latest_result["scores"], latest_result["labels"], latest_result["boxes"]):
            # Draw bounding box and label
            x_min, y_min, x_max, y_max = box.tolist()
            height, width, _ = frame.shape  # Get image dimensions
            x_min = int(x_min * width)
            y_min = int(y_min * height)
            x_max = int(x_max * width)
            y_max = int(y_max * height)
            object_name = model.config.id2label[label.item()]

            # Draw bounding box
            cv2.rectangle(frame, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 255, 0), 2)

            # Add label
            cv2.putText(frame, object_name, (int(x_min), int(y_min - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Display frame
    cv2.imshow('Object Detection', frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
