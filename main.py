import cv2
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_objectron = mp.solutions.objectron

cap = cv2.VideoCapture(0) # default_webcam= 0

with mp_objectron.Objectron(static_image_mode=True, max_num_objects=3, min_detection_confidence=0.5) as objectron:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            break

        # RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        results = objectron.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.detected_objects:
            for idx, detected_object in enumerate(results.detected_objects):
                # marcos, eixos e trajetória do objeto
                mp_drawing.draw_landmarks(image, detected_object.landmarks_2d, mp_objectron.BOX_CONNECTIONS)
                mp_drawing.draw_axis(image, detected_object.rotation, detected_object.translation)

                if len(trajectories) <= idx:
                    trajectories.append([])
                trajectories[idx].append(np.mean(detected_object.landmarks_2d.landmark, axis=0))

                for point in trajectories[idx]:
                    cv2.circle(image, (int(point.x * image.shape[1]), int(point.y * image.shape[0])), 5, (0, 255, 0),
                               -1)

        cv2.imshow('MediaPipe Objectron', image)
        if cv2.waitKey(5) & 0xFF == 27:
            break

    cap.release()




