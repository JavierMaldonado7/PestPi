import cv2
import requests
import numpy as np

url = "http://raspberry_pi_ip:5000/video_feed"
yolo_config_path = "/yolov3.cfg"
yolo_weights_path = "/yolov3.weights"
yolo_classes_path = "/coco.names"

# Load YOLO model and classes
net = cv2.dnn.readNet(yolo_weights_path, yolo_config_path)
with open(yolo_classes_path, 'r') as f:
    classes = [line.strip() for line in f.readlines()]

def process_frame(frame):
    height, width = frame.shape[:2]

    # Create a blob from the frame and set it as input to the network
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)

    # Get the names of the output layers
    layer_names = net.getUnconnectedOutLayersNames()

    # Run forward pass to get detections
    detections = net.forward(layer_names)

    # Loop over the detections and draw bounding boxes
    for detection in detections:
        for obj in detection:
            scores = obj[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.5:  # Set a confidence threshold
                center_x = int(obj[0] * width)
                center_y = int(obj[1] * height)
                w = int(obj[2] * width)
                h = int(obj[3] * height)

                # Calculate coordinates for the bounding box
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                # Draw bounding box and label on the frame
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                label = f"{classes[class_id]}: {confidence:.2f}"
                cv2.putText(frame, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow("Object Detection", frame)
    cv2.waitKey(1)

if __name__ == "__main__":
    cap = cv2.VideoCapture(url)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Process the frame (perform object detection)
        process_frame(frame)

    cap.release()
    cv2.destroyAllWindows()