import cv2
import tensorflow as tf
import timeit
import numpy as np
import matplotlib.pyplot as plt


def correct(img_in, k, d, dims):
    dim1 = img_in.shape[:2][::-1]
    assert dim1[0] / dim1[1] == dims[0] / dims[1], "Image to correct needs to have same aspect ratio as the ones used in calibration"
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(k, d, np.eye(3), k, dims, cv2.CV_16SC2)
    img_out = cv2.remap(img_in, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    return img_out

confidences = []

labels = ['nose', 'left eye', 'right eye', 'left ear', 'right ear',
          'left shoulder', 'right shoulder', 'left elbow', 'right elbow',
          'left wrist', 'right wrist', 'left hip', 'right hip',
          'left knee', 'right knee', 'left ankle', 'right ankle']

interpreter = tf.lite.Interpreter(model_path="../models/movenet_lightning.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

capture = cv2.VideoCapture("wide.mp4")
Dims = tuple(np.load('./parameters/Dims.npy'))
K = np.load('./parameters/K.npy')
D = np.load('./parameters/D.npy')
# 첫 프레임을 읽어서 웹캠의 정보를 알아내는 부분
ret, cam = capture.read()
cam = correct(cam, k=K, d=D, dims=Dims)
h, w, n = cam.shape
total_point = 0
total_right = 0



while True:
    ret, cam = capture.read()
    if ret:
        cam = correct(cam, k=K, d=D, dims=Dims)
        start_t = timeit.default_timer()
        frame = cam.copy()
        frame = cv2.resize(frame, (192, 192))
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_tensor = tf.convert_to_tensor(rgb_frame, dtype=tf.float32)
        frame_extended = tf.expand_dims(frame_tensor, axis=0)
        input_image = tf.cast(frame_extended, dtype=tf.float32)

        interpreter.set_tensor(input_details[0]['index'], input_image.numpy())
        interpreter.invoke()
        points = interpreter.get_tensor(output_details[0]['index'])

        for index, point in enumerate(points[0][0]):
            total_point += 1
            if point[2] > 0.3:
                total_right +=1
                cv2.circle(cam, (int(point[1] * w), (int(point[0] * h))), 3, (0, 255, 0))
                if index > 4:
                    cv2.putText(cam, org=(int(point[1] * w), (int(point[0] * h) - 10)), fontScale=0.3,
                                color=(0, 255, 0), text=labels[index], fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                lineType=cv2.LINE_AA, thickness=1)

        terminate_t = timeit.default_timer()
        fps = int(1. / (terminate_t - start_t))
        confidence = int(total_right/total_point*100)
        if len(confidences) < 500:
            confidences.append(confidence)
        cv2.putText(cam, org=(5,20), fontScale=0.5,
                    color=(0, 255, 0), text=str(fps), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    lineType=cv2.LINE_AA, thickness=1)

        cv2.putText(cam, org=(50, 20), fontScale=0.5,
                    color=(0, 255, 0), text='{}%'.format(str(confidence)), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    lineType=cv2.LINE_AA, thickness=1)

        cv2.imshow('camera', cam)

    if cv2.waitKey(1) & 0Xff == 27:
        break

plt.ylim((50, 110))
plt.plot(confidences)
plt.show()

capture.release()
cv2.destroyAllWindows()