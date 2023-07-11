
import cv2
import numpy as np
import tensorflow as tf
import BKNetStyle
from mtcnn.mtcnn import MTCNN
import CNN2Head_input
from const import *
from matplotlib import pyplot as plt
from imutils import paths
def load_network():
    sess = tf.compat.v1.Session()
    x = tf.compat.v1.placeholder(tf.float32, [None, 28, 28, 1])
    y_smile_conv, phase_train, keep_prob = BKNetStyle.BKNetModel(x)
    print("Load model")
    saver = tf.compat.v1.train.Saver()
    saver.restore(sess, './save/last/model.ckpt')
    return sess, x, y_smile_conv, phase_train, keep_prob

def load_test_img(base_dir):
    smile_train, smile_tests = CNN2Head_input.getSmileImage(BASE_DIR=base_dir)
    return smile_tests
def load_test_happy_img(base_dir):
    count = 0
    DATASET = base_dir + "dataset\happy"
    for imagePath in sorted(list(paths.list_images(DATASET))):
        image = cv2.imread(imagePath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, (28, 28))
        image = (image) / 255.0
        T = np.zeros([28, 28, 1])
        T[:, :, 0] = image
        test_img = []
        test_img.append(T)
        test_img = np.asarray(test_img)
        T = np.reshape(T, (-1, 28, 28, 1))
        predict_y_smile = sess.run(y_smile_conv, feed_dict={x: test_img, phase_train: False, keep_prob: 1.0})
        if np.argmax(predict_y_smile) == 0:
            print("Not smiling")
        if np.argmax(predict_y_smile) == 1:
            print("Smiling")
            count += 1
    print(count/len(list(paths.list_images(DATASET))))

def main(sess, x, y_smile_conv,  phase_train, keep_prob):
    # load_test_happy_img(BASE_DIR)
    smile_tests = load_test_img(BASE_DIR)
    smile_tests = smile_tests[3020:3040]
    detector = MTCNN()
    # plt show list 20 image
    
    plt.figure(figsize=(20, 20))
    plt.tight_layout()
    for index, smile_test in enumerate(smile_tests):
        plt.subplot(4, 5, index + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        if len(smile_test.shape) == 1:
            img = smile_test[0]
            img = (img) / 255.0
            T = np.zeros([28, 28, 1])
            T[:, :, 0] = img
            test_img = []
            test_img.append(T)
            test_img = np.asarray(test_img)
            T = np.reshape(T, (-1, 28, 28, 1))
            predict_y_smile_conv = sess.run(y_smile_conv, feed_dict={x: test_img, phase_train: False, keep_prob: 1.0})
            if np.argmax(predict_y_smile_conv) == 0:
                title = "Predict: not smiling" + f"\nActual: {smile_test[1]}"
            if np.argmax(predict_y_smile_conv) == 1:
                title = "Predict: smiling" + f"\nActual: {smile_test[1]}"
            plt.title(title, fontdict={'fontsize': "10"})
            plt.imshow(smile_test[0], cmap='gray')
        if len(smile_test.shape) == 3:
            result = detector.detect_faces(smile_test)
            if not result:
                print("No face detected")
                continue
            for r in result:
                face_position = r.get('box')
                x_coordinate = face_position[0]
                y_coordinate = face_position[1]
                w_coordinate = face_position[2]
                h_coordinate = face_position[3]
                img = smile_test[y_coordinate:y_coordinate + h_coordinate, x_coordinate:x_coordinate+w_coordinate]
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                img = cv2.resize(img, (28, 28))
                img = (img - 128) / 255.0
                T = np.zeros([28, 28, 1])
                T[:, :, 0] = img
                test_img = []
                test_img.append(T)
                test_img = np.asarray(test_img)
                T = np.reshape(T, (-1, 28, 28, 1))
                predict_y_smile_conv = sess.run(y_smile_conv, feed_dict={x: test_img, phase_train: False, keep_prob: 1})
                if np.argmax(predict_y_smile_conv) == 0:
                    title = "Not smiling"
                else:
                    title = "Smiling"
            plt.title(title, fontdict={'fontsize': "20"})
            plt.imshow(smile_test[0], cmap='gray')
    plt.show()
    

if __name__ == '__main__':
    sess, x, y_smile_conv, phase_train, keep_prob = load_network()
    main(sess, x, y_smile_conv, phase_train, keep_prob)