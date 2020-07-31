from keras import backend as K
import tensorflow as tf

'''
This file contains multiple loss functions with which we experimented with
'''

# Initializing foreground threshold (percentage of road pixels s.t. patch is considered road)
foreground_threshold = 0.5


def dice_loss(y_true, y_pred):
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])

    numerator = 2.0 * tf.reduce_sum(y_true_f * y_pred_f, axis=0)
    denominator = tf.reduce_sum(y_true_f + y_pred_f, axis=0)

    return 1.0 - (numerator + 1.0) / (denominator + 1.0)


def iou_coef(y_true, y_pred, smooth=1):
    intersection = tf.reduce_sum(tf.abs(y_true * y_pred), axis=[1, 2, 3])
    union = tf.reduce_sum(y_true, axis=[1, 2, 3]) + tf.reduce_sum(y_pred, axis=[1, 2, 3]) - intersection
    iou = tf.reduce_mean((intersection + smooth) / (union + smooth), axis=0)
    return iou


def iou_loss_log(y_true, y_pred):
    intersection = tf.reduce_sum(y_true * y_pred, axis=[1, 2, 3])
    union = tf.reduce_sum(y_true + y_pred, axis=[1, 2, 3]) - intersection
    iou = tf.reduce_mean(intersection / union, axis=0)
    return -tf.math.log(iou)


def patch_f1_score(y_true, y_pred):
    # Initializing ones kernel for summing up patch mean
    ones = tf.ones([16, 16, 1, 1])

    y_pred_disc = tf.where(tf.greater(y_pred, 0.5), 1.0, 0.0)

    # Taking mean of 16x16 patches and normalizing result
    y_true_patch_mean = (1.0/256.0) * tf.nn.conv2d(y_true, ones, strides=[1, 16, 16, 1], padding='VALID')
    y_pred_patch_mean = (1.0/256.0) * tf.nn.conv2d(y_pred_disc, ones, strides=[1, 16, 16, 1], padding='VALID')

    # Converting patch means to either zero or one, depending on foreground threshold
    y_true_patch_label = tf.where(tf.greater(y_true_patch_mean, foreground_threshold), 1.0, 0.0)
    y_pred_patch_label = tf.where(tf.greater(y_pred_patch_mean, foreground_threshold), 1.0, 0.0)

    y_true_patch_label = tf.reshape(y_true_patch_label, [-1])
    y_pred_patch_label = tf.reshape(y_pred_patch_label, [-1])

    TP = tf.math.count_nonzero(y_pred_patch_label * y_true_patch_label)
    TN = tf.math.count_nonzero((y_pred_patch_label - 1.0) * (y_true_patch_label - 1.0))
    FP = tf.math.count_nonzero(y_pred_patch_label * (y_true_patch_label - 1.0))
    FN = tf.math.count_nonzero((y_pred_patch_label - 1.0) * y_true_patch_label)

    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1 = 2.0 * precision * recall / (precision + recall)
    return f1


def patch_loss(y_true, y_pred):
    # Initializing ones kernel for summing up patch mean
    ones = tf.ones([16, 16, 1, 1])

    y_pred_disc = tf.where(tf.greater(y_pred, 0.5), 1.0, 0.0)

    # Taking mean of 16x16 patches and normalizing result
    y_true_patch_mean = (1.0/256.0) * tf.nn.conv2d(y_true, ones, strides=[1, 16, 16, 1], padding='VALID')
    y_pred_patch_mean = (1.0/256.0) * tf.nn.conv2d(y_pred_disc, ones, strides=[1, 16, 16, 1], padding='VALID')

    # Converting patch means to either zero or one, depending on foreground threshold
    y_true_patch_label = tf.where(tf.greater(y_true_patch_mean, foreground_threshold), 1.0, 0.0)
    y_pred_patch_label = tf.where(tf.greater(y_pred_patch_mean, foreground_threshold), 1.0, 0.0)

    return tf.reduce_mean(tf.square(y_true_patch_label - y_pred_patch_label))


#https://lars76.github.io/neural-networks/object-detection/losses-for-segmentation/
def weighted_cross_entropy(beta):
    def convert_to_logits(y_pred):
        # see https://github.com/tensorflow/tensorflow/blob/r1.10/tensorflow/python/keras/backend.py#L3525
        y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon())
        return tf.math.log(y_pred / (1 - y_pred))

    def loss(y_true, y_pred):
        y_pred = convert_to_logits(y_pred)
        loss = tf.nn.weighted_cross_entropy_with_logits(logits=y_pred, labels=y_true, pos_weight=beta)
        # or reduce_sum and/or axis=-1
        return tf.reduce_mean(loss)

    return loss
