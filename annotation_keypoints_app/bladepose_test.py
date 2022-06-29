import cv2
import mediapipe as mp
import timeit
import time

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

cap = cv2.VideoCapture("../videos/taebo.mp4")

with mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        model_complexity=2) as pose:
    while cap.isOpened():

        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            time.sleep(2)
            break
        start_t = timeit.default_timer()
        #
        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.

        image = image[500:-500, 0:-1]
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(image)

        # Draw the pose annotation on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        mp_drawing.draw_landmarks(
            image,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
        # Flip the image horizontally for a selfie-view display.

        terminate_t = timeit.default_timer()
        fps = int(1. / (terminate_t - start_t))

        image = cv2.flip(image, 1)

        cv2.putText(image, org=(5, 20), fontScale=0.5,
                    color=(0, 255, 0), text=str(fps), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    lineType=cv2.LINE_AA, thickness=1)

        image = cv2.resize(image, (int(image.shape[1]/2), int(image.shape[0]/2)))
        cv2.imshow('MediaPipe Pose', image)

        if cv2.waitKey(5) & 0xFF == 27:
            break
cap.release()
