import cv2
import numpy as np
import time

# Start the webcam
cap = cv2.VideoCapture(0)
time.sleep(3)  # Allow the camera to adjust

# Capture the background
background = None
for i in range(50):
    ret, background = cap.read()

# Flip the background for consistency
background = cv2.flip(background, 1)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame for consistency
    frame = cv2.flip(frame, 1)

    # Convert the frame to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define HSV range for the cloak color (Red)
    lower_red1 = np.array([0, 120, 70])
    upper_red1 = np.array([10, 255, 255])

    lower_red2 = np.array([170, 120, 70])
    upper_red2 = np.array([180, 255, 255])

    # Create masks to detect red color
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)

    mask = mask1 + mask2  # Combine both masks

    # Refine the mask using morphological operations
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, np.ones((3, 3), np.uint8))

    # Create an inverse mask
    mask_inv = cv2.bitwise_not(mask)

    # Replace the detected red area with the background
    res1 = cv2.bitwise_and(background, background, mask=mask)
    res2 = cv2.bitwise_and(frame, frame, mask=mask_inv)

    # Combine both results
    final_output = cv2.addWeighted(res1, 1, res2, 1, 0)

    # Display the output
    cv2.imshow("Invisible Cloak", final_output)

    # Exit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
