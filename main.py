
import cv2
import numpy as np
import tensorflow as tf
import BKNetStyle
from const import *
from mtcnn.mtcnn import MTCNN


def load_network():
    sess = tf.compat.v1.Session()
    x = tf.compat.v1.placeholder(tf.float32, [None, 28, 28, 1])
    y_smile_conv, phase_train, keep_prob = BKNetStyle.BKNetModel(x)
    print('Restore model')
    saver = tf.compat.v1.train.Saver()
    saver.restore(sess, './save/exp/model.ckpt')
    print('OK')
    return sess, x, y_smile_conv, phase_train, keep_prob


def draw_label(image, x, y, w, h, label, font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=1, thickness=2):
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 155, 255), 2)
    cv2.putText(image, label, (x, y), font, font_scale, (255, 255, 255), thickness)


def main(sess, x, y_smile_conv,  phase_train, keep_prob):
    # capture video
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1080)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    detector = MTCNN()

    while True:
        # get video frame
        ret, img = cap.read()

        if not ret:
            print("error: failed to capture image")
            return -1

        # detect face and crop face, convert to gray, resize to 48x48
        original_img = img
        result = detector.detect_faces(original_img)
        if not result:
            cv2.imshow("result", original_img)
            continue
        face_position = result[0].get('box')
        x_coordinate = face_position[0]
        y_coordinate = face_position[1]
        w_coordinate = face_position[2]
        h_coordinate = face_position[3]
        img = original_img[y_coordinate:y_coordinate +
                           h_coordinate, x_coordinate:x_coordinate+w_coordinate]
        if(img.size == 0):
            cv2.imshow("result", original_img)
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (28, 28))
        img = (img - 128) / 255.0
        T = np.zeros([28, 28, 1])
        T[:, :, 0] = img
        test_img = []
        test_img.append(T)
        test_img = np.asarray(test_img)

        T = np.reshape(T, (-1, 28, 28, 1))

        predict_y_smile_conv = sess.run(y_smile_conv, feed_dict={
                                        x: test_img, phase_train: False, keep_prob: 1})

        smile_label = "not miling" if np.argmax(
            predict_y_smile_conv) == 0 else "smiling"

        label = "{}".format(smile_label)
        draw_label(original_img, x_coordinate, y_coordinate,
                   w_coordinate, h_coordinate, label)

        cv2.imshow("result", original_img)
        key = cv2.waitKey(1)

        if key == 27:
            break


if __name__ == '__main__':
    sess, x, y_smile_conv, phase_train, keep_prob = load_network()
    main(sess, x, y_smile_conv, phase_train, keep_prob)
