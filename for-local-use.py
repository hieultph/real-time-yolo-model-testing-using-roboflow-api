import cv2
import torch

# Load the YOLO model
model = torch.hub.load('ultralytics/yolov10', 'custom', path='./ai_best.pt', source='local')  # Replace 'best.pt' with your model file

# Open a connection to the webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    
    if not ret:
        break
    
    # Make predictions on the frame
    results = model(frame)

    # Render the results on the frame
    frame_with_results = results.render()[0]  # results.render() returns a list, we use the first element

    # Display the frame with predictions
    cv2.imshow('YOLO Prediction', frame_with_results)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()