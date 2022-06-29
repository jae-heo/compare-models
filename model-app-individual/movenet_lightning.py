import cv2
import tensorflow as tf
import timeit

labels = ['nose', 'left eye', 'right eye', 'left ear', 'right ear',
          'left shoulder', 'right shoulder', 'left elbow', 'right elbow',
          'left wrist', 'right wrist', 'left hip', 'right hip',
          'left knee', 'right knee', 'left ankle', 'right ankle']

interpreter = tf.lite.Interpreter(model_path="../models/movenet_lightning.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

capture = cv2.VideoCapture(0)

# 첫 프레임을 읽어서 웹캠의 정보를 알아내는 부분
ret, cam = capture.read()
h, w, n = cam.shape

while True:
    ret, cam = capture.read()
    if ret:
        start_t = timeit.default_timer()

        frame = cam.copy()
        frame = cv2.resize(cam, (192, 192))
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_tensor = tf.convert_to_tensor(rgb_frame, dtype=tf.float32)
        frame_extended = tf.expand_dims(frame_tensor, axis=0)
        input_image = tf.cast(frame_extended, dtype=tf.float32)

        interpreter.set_tensor(input_details[0]['index'], input_image.numpy())
        interpreter.invoke()
        points = interpreter.get_tensor(output_details[0]['index'])

        for index, point in enumerate(points[0][0]):
            if point[2] > 0.3:
                cv2.circle(cam, (int(point[1] * w), (int(point[0] * h))), 3, (0, 255, 0))
                if index > 4:
                    cv2.putText(cam, org=(int(point[1] * w), (int(point[0] * h) - 10)), fontScale=0.3,
                                color=(0, 255, 0), text=labels[index], fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                lineType=cv2.LINE_AA, thickness=1)

        terminate_t = timeit.default_timer()
        fps = int(1. / (terminate_t - start_t))
        cv2.putText(cam, org=(5,20), fontScale=0.5,
                    color=(0, 255, 0), text=str(fps), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    lineType=cv2.LINE_AA, thickness=1)
        cv2.imshow('camera', cam)

    if cv2.waitKey(1) & 0Xff == 27:
        break

capture.release()
cv2.destroyAllWindows()

#
# image_path = './squat.jpg'
#
# img = cv2.imread(image_path, cv2.IMREAD_COLOR).copy()
# img = cv2.resize(img, (256, 256))
# rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# image = tf.convert_to_tensor(rgb, dtype=tf.float32)
# image = tf.expand_dims(image, axis=0)
#
# interpreter = tf.lite.Interpreter(model_path="./movenet_thunder.tflite")
# interpreter.allocate_tensors()
#
# input_image = tf.cast(image, dtype=tf.float32)
# input_details = interpreter.get_input_details()
# output_details = interpreter.get_output_details()
#
# interpreter.set_tensor(input_details[0]['index'], input_image.numpy())
#
# interpreter.invoke()
#
# points = interpreter.get_tensor(output_details[0]['index'])
#
# img = cv2.imread(image_path, cv2.IMREAD_COLOR).copy()
#
# h, w, n = img.shape
#
# for point in points[0][0]:
#     cv2.circle(img, (int(point[1] * w), (int(point[0] * h))), 3, (0, 255, 0))
#
