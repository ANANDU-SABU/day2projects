import cv2
try:
    import mediapipe as mp
except ImportError as e:
    raise ImportError(
        "mediapipe is not installed. Install it with:\n"
        "  pip install mediapipe\n"
    ) from e

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1)  # Single face for stable point

# Open webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert BGR → RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:
        for face in results.multi_face_landmarks:

            # Landmark index 10 = HEAD CENTER / TOP
            head_center = face.landmark[10]

            h, w, _ = frame.shape
            cx = int(head_center.x * w)
            cy = int(head_center.y * h)

            # Draw red dot
            cv2.circle(frame, (cx, cy), 8, (0, 0, 255), -1)

    cv2.imshow("Head Center", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break  # ESC to exit

cap.release()
cv2.destroyAllWindows()
