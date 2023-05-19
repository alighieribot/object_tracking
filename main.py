import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt

mp_drawing = mp.solutions.drawing_utils
mp_face_detection = mp.solutions.face_detection

cap = cv2.VideoCapture(0)  # default_webcam= 0
trajectories = []

with mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            break

        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = face_detection.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.detections:
            for idx, detection in enumerate(results.detections):
                mp_drawing.draw_detection(image, detection)

                box = detection.location_data.relative_bounding_box
                center = np.array([box.xmin + box.width / 2, box.ymin + box.height / 2])

                if len(trajectories) <= idx:
                    trajectories.append([])

                trajectories[idx].append(center)

                for point in trajectories[idx]:
                    cv2.circle(image, (int(point[0] * image.shape[1]), int(point[1] * image.shape[0])), 5, (0, 255, 0), -1)

        cv2.imshow('MediaPipe Face Detection', image)
        if cv2.waitKey(5) & 0xFF == 27:
            break

    cap.release()

# plot
for idx, trajectory in enumerate(trajectories):
    trajectory = np.array(trajectory)
    plt.plot(trajectory[:, 0], trajectory[:, 1])
    plt.title(f'Trajetória do Rosto {idx+1} (2D)')
    plt.show()
