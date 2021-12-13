import tensorflow as tf

def create_convnet(input_shape=[645, 8], nb_classes=19):
    model = tf.keras.Sequential(name='golfvgg')

    # block1
    model.add(tf.keras.layers.Conv1D(64, 3, strides=1, padding='same', activation='relu', input_shape=input_shape, name='block1_conv1'))
    model.add(tf.keras.layers.Conv1D(64, 3, strides=1, padding='same', activation='relu', name='block1_conv2'))
    model.add(tf.keras.layers.MaxPool1D(pool_size=2, strides=2, name='block1_pool'))

    # block2
    model.add(tf.keras.layers.Conv1D(128, 3, strides=1, padding='same', activation='relu', name='block2_conv1'))
    model.add(tf.keras.layers.Conv1D(128, 3, strides=1, padding='same', activation='relu', name='block2_conv2'))
    model.add(tf.keras.layers.MaxPool1D(pool_size=2, strides=2, name='block2_pool'))

    # block3
    model.add(tf.keras.layers.Conv1D(256, 3, strides=1, padding='same', activation='relu', name='block3_conv1'))
    model.add(tf.keras.layers.Conv1D(256, 3, strides=1, padding='same', activation='relu', name='block3_conv2'))
    model.add(tf.keras.layers.Conv1D(256, 3, strides=1, padding='same', activation='relu', name='block3_conv3'))
    model.add(tf.keras.layers.Conv1D(256, 3, strides=1, padding='same', activation='relu', name='block3_conv4'))
    model.add(tf.keras.layers.MaxPool1D(pool_size=2, strides=2, name='block3_pool'))

    # block4
    model.add(tf.keras.layers.Conv1D(512, 3, strides=1, padding='same', activation='relu', name='block4_conv1'))
    model.add(tf.keras.layers.Conv1D(512, 3, strides=1, padding='same', activation='relu', name='block4_conv2'))
    model.add(tf.keras.layers.Conv1D(512, 3, strides=1, padding='same', activation='relu', name='block4_conv3'))
    model.add(tf.keras.layers.Conv1D(512, 3, strides=1, padding='same', activation='relu', name='block4_conv4'))
    model.add(tf.keras.layers.MaxPool1D(pool_size=2, strides=2, name='block4_pool'))

    # block5
    model.add(tf.keras.layers.Conv1D(512, 3, strides=1, padding='same', activation='relu', name='block5_conv1'))
    model.add(tf.keras.layers.Conv1D(512, 3, strides=1, padding='same', activation='relu', name='block5_conv2'))
    model.add(tf.keras.layers.Conv1D(512, 3, strides=1, padding='same', activation='relu', name='block5_conv3'))
    model.add(tf.keras.layers.Conv1D(512, 3, strides=1, padding='same', activation='relu', name='block5_conv4'))
    model.add(tf.keras.layers.MaxPool1D(pool_size=2, strides=2, name='block5_pool'))

    # top layers
    model.add(tf.keras.layers.Flatten(name='flatten'))
    model.add(tf.keras.layers.Dense(256, activation='relu', name='fc1'))
    model.add(tf.keras.layers.Dense(nb_classes, name='predictions'))

    return model
