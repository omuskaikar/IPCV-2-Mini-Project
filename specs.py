import cv2
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

# Load the spectacles image
specs_image = cv2.imread("specs.jpg", cv2.IMREAD_UNCHANGED)  # Ensure the image has an alpha channel

# Add alpha channel if missing
if specs_image.shape[2] == 3:  # If no alpha channel
    alpha_channel = np.ones(specs_image.shape[:2], dtype=np.uint8) * 255  # Fully opaque
    specs_image = np.dstack((specs_image, alpha_channel))

def overlay_image(background, overlay, position):
    """
    Overlay an RGBA image on top of a BGR background at the specified position.
    If the overlay image does not have an alpha channel, apply it directly.
    """
    x, y, w, h = position
    overlay_resized = cv2.resize(overlay, (w, h), interpolation=cv2.INTER_AREA)

    # Ensure alpha channel is respected
    for i in range(overlay_resized.shape[0]):
        for j in range(overlay_resized.shape[1]):
            if overlay_resized[i, j, 3] != 0:  # Check transparency
                background[y + i, x + j] = overlay_resized[i, j, :3]  # Apply RGB values

cap = cv2.VideoCapture(0)
with mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as face_mesh:

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        frame.flags.writeable = False
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(frame_rgb)

        frame.flags.writeable = True
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Get landmarks for left and right eyes
                left_eye = [face_landmarks.landmark[i] for i in range(33, 133)]
                right_eye = [face_landmarks.landmark[i] for i in range(263, 362)]

                # Calculate bounding boxes for both eyes
                def get_bbox(landmarks):
                    x = [lm.x for lm in landmarks]
                    y = [lm.y for lm in landmarks]
                    xmin, xmax = int(min(x) * frame.shape[1]), int(max(x) * frame.shape[1])
                    ymin, ymax = int(min(y) * frame.shape[0]), int(max(y) * frame.shape[0])
                    return xmin, ymin, xmax - xmin, ymax - ymin

                left_bbox = get_bbox(left_eye)
                right_bbox = get_bbox(right_eye)

                # Overlay spectacles
                overlay_image(frame_bgr, specs_image, left_bbox)
                overlay_image(frame_bgr, specs_image, right_bbox)

        # Show the frame
        cv2.imshow('Virtual Try-On', cv2.flip(frame_bgr, 1))
        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
