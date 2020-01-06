import tensorflow as tf

IMG_HEIGHT = 1200
IMG_WIDTH = 512

def dice_metric(y_true, y_pred):
  from tensorflow.keras import backend as K 
  reshape = lambda x: K.reshape(x, (-1, IMG_HEIGHT * IMG_WIDTH, 1))
  dice = lambda x: 2 * K.sum(x[0] * x[1], axis=1) / (K.sum(x[0], axis=1) + K.sum(x[1], axis=1))
  y_true_flat = reshape(y_true)
  y_pred_flat = reshape(y_pred)
  y_true_flat_bi = tf.where(tf.greater_equal(y_true_flat, 0.5), y_true_flat, tf.zeros_like(y_true_flat))
  y_pred_flat_bi = tf.where(tf.greater_equal(y_pred_flat, 0.5), y_pred_flat, tf.zeros_like(y_pred_flat))
  _dice = dice([y_true_flat_bi, y_pred_flat_bi])
  return K.mean(_dice)

def convert_to_logits(y_pred):
  y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon())
  return tf.log(y_pred / (1 - y_pred))

def weighted_cross_entropy(y_true, y_pred):
    y_pred = convert_to_logits(y_pred)
    loss = tf.nn.weighted_cross_entropy_with_logits(logits=y_pred, targets=y_true, pos_weight=tf.constant([[0.75,  0.25]]))
    # or reduce_sum and/or axis=-1
    return tf.reduce_mean(loss)


def unet_original_size(input_size = (1200,512,1)):
  input_img = tf.keras.layers.Input(input_size)
  conv1 = tf.keras.layers.Conv2D(64, 3, activation = tf.nn.relu, padding = 'same')(input_img)
  conv1 = tf.keras.layers.Conv2D(64, 3, activation = tf.nn.relu, padding = 'same')(conv1)

  pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv1)
  conv2 = tf.keras.layers.Conv2D(128, 3, activation = tf.nn.relu, padding = 'same')(pool1)
  conv2 = tf.keras.layers.Conv2D(128, 3, activation = tf.nn.relu, padding = 'same')(conv2)

  pool2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv2)
  conv3 = tf.keras.layers.Conv2D(256, 3, activation = tf.nn.relu, padding = 'same')(pool2)
  conv3 = tf.keras.layers.Conv2D(256, 3, activation = tf.nn.relu, padding = 'same')(conv3)

  pool3 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv3)
  conv4 = tf.keras.layers.Conv2D(512, 3, activation = tf.nn.relu, padding = 'same')(pool3)
  conv4 = tf.keras.layers.Conv2D(512, 3, activation = tf.nn.relu, padding = 'same')(conv4)

  #drop4 = tf.keras.layers.Dropout(0)(conv4)

  pool4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv4)
  conv5 = tf.keras.layers.Conv2D(1024, 3, activation = tf.nn.relu, padding = 'same')(pool4)
  conv5 = tf.keras.layers.Conv2D(1024, 3, activation = tf.nn.relu, padding = 'same')(conv5)

  #drop5 = tf.keras.layers.Dropout(0)(conv5)
  up6 = tf.keras.layers.UpSampling2D(size = (2,2))(conv5)
  before6 = tf.keras.layers.Conv2D(512, 2, activation = tf.nn.relu, padding = 'same')(up6)
  merge6 = tf.keras.layers.concatenate([conv4,before6], axis = 3)
  conv6 = tf.keras.layers.Conv2D(512, 3, activation = tf.nn.relu, padding = 'same')(merge6)
  conv6 = tf.keras.layers.Conv2D(512, 3, activation = tf.nn.relu, padding = 'same')(conv6)

  up7 = tf.keras.layers.UpSampling2D(size = (2,2))(conv6)
  before7 = tf.keras.layers.Conv2D(256, 2, activation = tf.nn.relu, padding = 'same')(up7)
  merge7 = tf.keras.layers.concatenate([conv3,before7], axis = 3)
  conv7 = tf.keras.layers.Conv2D(256, 3, activation = tf.nn.relu, padding = 'same')(merge7)
  conv7 = tf.keras.layers.Conv2D(256, 3, activation = tf.nn.relu, padding = 'same')(conv7)

  up8 = tf.keras.layers.UpSampling2D(size = (2,2))(conv7)
  before8 = tf.keras.layers.Conv2D(128, 2, activation = tf.nn.relu, padding = 'same')(up8)
  merge8 = tf.keras.layers.concatenate([conv2,before8], axis = 3)
  conv8 = tf.keras.layers.Conv2D(128, 3, activation = tf.nn.relu, padding = 'same')(merge8)
  conv8 = tf.keras.layers.Conv2D(128, 3, activation = tf.nn.relu, padding = 'same')(conv8)

  up9 = tf.keras.layers.UpSampling2D(size = (2,2))(conv8)
  before9 = tf.keras.layers.Conv2D(64, 2, activation = tf.nn.relu, padding = 'same')(up9)
  merge9 = tf.keras.layers.concatenate([conv1,up9], axis = 3)
  conv9 = tf.keras.layers.Conv2D(64, 3, activation = tf.nn.relu, padding = 'same')(merge9)
  conv9 = tf.keras.layers.Conv2D(64, 3, activation = tf.nn.relu, padding = 'same')(conv9)
  conv9 = tf.keras.layers.Conv2D(16, 3, activation = tf.nn.relu, padding = 'same')(conv9)
  conv9 = tf.keras.layers.Conv2D(16, 1, activation = tf.nn.relu, padding = 'same')(conv9)
  conv10 = tf.keras.layers.Conv2D(1, 1, activation = tf.nn.sigmoid)(conv9)

  model = tf.keras.Model(inputs = input_img, outputs = conv10)
  #model.compile(optimizer = tf.keras.optimizers.Adam(lr = 1e-4), loss = tf.keras.losses.binary_crossentropy, metrics = [tf.keras.metrics.Accuracy(),dice_coef_loss,tf.keras.metrics.MeanIoU(num_classes=2)])
  model.compile(optimizer = tf.keras.optimizers.Adam(lr = 1e-4), loss = weighted_cross_entropy, metrics = [tf.keras.metrics.Accuracy(),dice_metric])

  #model.summary()

  return model