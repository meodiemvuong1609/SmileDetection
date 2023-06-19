from pathlib import Path
import CNN2Head_input
import os
import tensorflow as tf
import numpy as np
import BKNetStyle
from const import *
import sys

sys.setrecursionlimit(150000)
sess = tf.compat.v1.Session()
sess.as_default()
tf.compat.v1.disable_eager_execution()
np_load_old = np.load
np.load = lambda *a, **k: np_load_old(*a, allow_pickle=True, **k)
smile_train, smile_test = CNN2Head_input.getSmileImage()

def one_hot(index, num_classes):
    tmp = np.zeros(num_classes, dtype=np.float32)
    if index == "smiling":
        index = 1
    else:
        index = 0
        tmp[index] = 1.0
    tmp[index] = 1.0
    return tmp


global_step = tf.compat.v1.train.get_or_create_global_step()

x, y_, mask = BKNetStyle.Input()

y_smile_conv, phase_train, keep_prob = BKNetStyle.BKNetModel(x)


smile_loss, l2_loss, loss = BKNetStyle.selective_loss(y_smile_conv, y_, mask)


train_step = BKNetStyle.train_op(loss, global_step)

smile_mask = tf.compat.v1.get_collection("smile_mask")[0]


y_smile =  tf.compat.v1.get_collection("y_smile")[0]


smile_correct_prediction = tf.equal(
    tf.argmax(y_smile_conv, 1), tf.argmax(y_smile, 1))


smile_true_pred = tf.reduce_sum(
    tf.cast(smile_correct_prediction, dtype=tf.float32) * smile_mask)


train_data = []
# Mask: Smile -> 0
for i in range(len(smile_train) * 10):
    img = (smile_train[i % 3000][0] - 128) / 255.0
    label = (smile_train[i % 3000][1])
    train_data.append((img, one_hot(label, 4), 0.0))


def init_save_model():
    saver = tf.compat.v1.train.Saver()

    Path(SAVE_FOLDER).mkdir(parents=True, exist_ok=True)

    if not os.path.isfile(SAVE_FOLDER + "model.ckpt.index"):
        print(SAVE_FOLDER)
        print("Create new model")
        sess.run(tf.compat.v1.global_variables_initializer())
        print("OK")
    else:
        print("Restoring existed model")
        saver.restore(sess, SAVE_FOLDER + "model.ckpt")
        print("OK")
    return saver


def init_summary():
    loss_summary_placeholder = tf.compat.v1.placeholder(tf.float32)
    tf.summary.scalar("loss", loss_summary_placeholder)
    merge_summary = tf.compat.v1.summary.merge_all()
    writer = tf.compat.v1.summary.FileWriter("./summary/")
    return merge_summary, writer


def train():
    saver = init_save_model()
    loss_summary_placeholder = tf.compat.v1.placeholder(tf.float32)
    tf.compat.v1.summary.scalar("loss", loss_summary_placeholder)
    merge_summary = tf.compat.v1.summary.merge_all()
    writer = tf.compat.v1.summary.FileWriter("./summary/")

    learning_rate = tf.compat.v1.get_collection("learning_rate")[0]

    current_epoch = (int)(global_step.eval(session=sess) / (len(train_data) // BATCH_SIZE))
    for epoch in range(current_epoch + 1, NUM_EPOCHS):
        print("Epoch:", str(epoch))
        np.random.shuffle(train_data)
        train_img = []
        train_label = []
        train_mask = []

        for i in range(len(train_data)):
            train_img.append(train_data[i][0])
            train_label.append(train_data[i][1])
            train_mask.append(train_data[i][2])

        number_batch = len(train_data) // BATCH_SIZE

        avg_ttl = []
        avg_rgl = []
        avg_smile_loss = []

        smile_nb_true_pred = 0

        smile_nb_train = 0

        print("Learning rate: %f" % learning_rate.eval(session=sess))
        for batch in range(number_batch):
            top = batch * BATCH_SIZE
            bot = min((batch + 1) * BATCH_SIZE, len(train_data))
            batch_img = np.asarray(train_img[top:bot])
            batch_label = np.asarray(train_label[top:bot])
            batch_mask = np.asarray(train_mask[top:bot])

            for i in range(BATCH_SIZE):
                if batch_mask[i] == 0.0:
                    smile_nb_train += 1

            batch_img = CNN2Head_input.augmentation(batch_img, 28)
            batch_img = np.reshape(batch_img, (-1, 28, 28, 1))

            ttl, sml, l2l, _ = sess.run([loss, smile_loss, l2_loss, train_step],
                                        feed_dict={x: batch_img, y_: batch_label, mask: batch_mask,
                                                   phase_train: True,
                                                   keep_prob: 0.5})

            smile_nb_true_pred += sess.run(smile_true_pred, feed_dict={x: batch_img, y_: batch_label, mask: batch_mask,
                                                                       phase_train: True,
                                                                       keep_prob: 0.5})

            avg_ttl.append(ttl)
            avg_smile_loss.append(sml)

            avg_rgl.append(l2l)

        smile_train_accuracy = smile_nb_true_pred * 1.0 / smile_nb_train

        avg_smile_loss = np.average(avg_smile_loss)

        avg_rgl = np.average(avg_rgl)
        avg_ttl = np.average(avg_ttl)

        summary = sess.run(merge_summary, feed_dict={
            loss_summary_placeholder: avg_ttl})
        writer.add_summary(summary, global_step=epoch)

        print("Smile task train accuracy: " + str(smile_train_accuracy * 100))

        print("Total loss: " + str(avg_ttl) + ". L2-loss: " + str(avg_rgl))
        print("Smile loss: " + str(avg_smile_loss))

        saver.save(sess, SAVE_FOLDER + "model.ckpt")


if __name__ == "__main__":
    train()
