from keras import backend as K
import tensorflow as tf

# Initializing foreground threshold (percentage of road pixels s.t. patch is considered road)
foreground_threshold = 0.25

def dice_coef_old(y_true, y_pred, smooth=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2.0 * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def iou_coef_old(y_true, y_pred, smooth=1):
    intersection = K.sum(K.abs(y_true * y_pred), axis=[1, 2, 3])
    union = K.sum(y_true, [1, 2, 3]) + K.sum(y_pred, [1, 2, 3]) - intersection
    iou = K.mean((intersection + smooth) / (union + smooth), axis=0)
    return iou

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

def soft_dice_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred)

def dice_coef(y_true, y_pred, smooth=1):
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f, axis=0)
    dice = (2.0 * intersection + smooth) / (tf.reduce_sum(y_true_f, axis=0) + tf.reduce_sum(y_pred_f,  axis=0) + smooth)
    return dice

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

"""
    tf.extract_image_patches(images=y_true,
                             sizes=[1, 3, 3, 1],
                             strides=[1, 5, 5, 1],
                             rates=[1, 2, 2, 1],
                             padding='VALID')
"""
