import cv2
import tensorflow as tf
import timeit
import argparse


def run(model_name, video_name, log):
    labels = ['nose', 'left eye', 'right eye', 'left ear', 'right ear',
              'left shoulder', 'right shoulder', 'left elbow', 'right elbow',
              'left wrist', 'right wrist', 'left hip', 'right hip',
              'left knee', 'right knee', 'left ankle', 'right ankle']

    model_path = '../models/' + model_name
    video_path = '../videos/' + video_name

    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    capture = cv2.VideoCapture(video_path)
    ret, cam = capture.read()
    h, w, n = cam.shape

    while True:
        ret, cam = capture.read()
        if ret:
            start_t = timeit.default_timer()

            cam = cam[500:-500, 0:-1]

            frame = cam.copy()
            frame = cv2.resize(frame, (input_details[0]['shape'][1], input_details[0]['shape'][2]))
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_tensor = tf.convert_to_tensor(rgb_frame, dtype=tf.float32)
            frame_extended = tf.expand_dims(frame_tensor, axis=0)
            input_image = tf.cast(frame_extended, dtype=tf.float32)

            interpreter.set_tensor(input_details[0]['index'], input_image.numpy())
            interpreter.invoke()
            points = interpreter.get_tensor(output_details[0]['index'])

            for index, point in enumerate(points[0][0]):
                if point[2] > 0.4:
                    cv2.circle(cam, (int(point[1] * w), (int(point[0] * h))), 3, (0, 255, 0))
                    if index > 4:
                        cv2.putText(cam, org=(int(point[1] * w), (int(point[0] * h) - 10)), fontScale=0.3,
                                    color=(0, 255, 0), text=labels[index], fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                    lineType=cv2.LINE_AA, thickness=1)

            terminate_t = timeit.default_timer()
            fps = int(1. / (terminate_t - start_t))
            cv2.putText(cam, org=(5, 20), fontScale=0.5,
                        color=(0, 255, 0), text=str(fps), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        lineType=cv2.LINE_AA, thickness=1)

            h, w, n = cam.shape

            cv2.imshow('camera', cam)

        if cv2.waitKey(1) & 0Xff == 27:
            break

    capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_name", dest="model_name", action="store", nargs="?", default="movenet_thunder.tflite")
    parser.add_argument("-l", "--print_log", dest="log", action="store_true")
    parser.add_argument("-v", "--video_name", dest="video_name", action="store", nargs="?", default="yoga.mp4")
    args = parser.parse_args()

    run(args.model_name, args.video_name, args.log)

else:
    exit(123)
