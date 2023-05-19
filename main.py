import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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

    # trajectories
    trajectories = []
    if len(trajectories) <= idx:
        trajectories.append([])
    trajectories[idx].append(detected_object.translation)

    for idx, trajectory in enumerate(trajectories):
        trajectory = np.array(trajectory)

        # 2D
        plt.figure()
        plt.plot(trajectory[:, 0], trajectory[:, 1])
        plt.title(f'Trajetória do Objeto {idx + 1} (2D)')
        plt.xlabel('x')
        plt.ylabel('y')

        # 3D
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2])
        ax.set_title(f'Trajetória do Objeto {idx + 1} (3D)')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')

    plt.show(block=True)

    cap.release()





