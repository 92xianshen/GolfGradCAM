import os
import numpy as np
import tensorflow as tf
from scipy import interpolate as spinterp
import matplotlib.pyplot as plt

task = ('All', '450') # 

save_root = 'multi-time/{}'.format(task[1]) # save folder
if not os.path.exists(save_root):
    os.makedirs(save_root)

# Sensor and length selections
sensor_set = {
    'All'   : [0, 1, 2, 3, 4, 5, 6, 7], # SG + Acc. + Gyro.
    'SG'    : [0, 1],                   # SG
    'Acc.'  : [2, 3, 4],                # Acc.
    'Gyro.' : [5, 6, 7],                # Gyro.
}

sample_set = {
    '645': [350, 995], # 645
    '600': [350, 950], # 600
    '550': [350, 900], # 550
    '500': [350, 850], # 500
    '450': [350, 800], # 450
}

sensor, sample = sensor_set[task[0]], sample_set[task[1]]

nb_classes = 19
SHAPE = [sample[1] - sample[0], len(sensor)]
swing_length = sample[1] - sample[0]

def target_category_loss(x, category_index, nb_classes):
    return tf.multiply(x, tf.one_hot([category_index], nb_classes))

def target_category_loss_output_shape(input_shape):
    return input_shape

def normalize(x):
    return x / (tf.sqrt(tf.reduce_mean(tf.square(x))) + 1e-5)

def norm(x):
    return (x - x.min()) / (x.max() - x.min() + 1e-5)

@tf.custom_gradient
def custom_relu(x):
    y = tf.nn.relu(x)
    def grad(dy):
        dtype = x.dtype
        return dy * tf.cast(dy > 0., dtype) * tf.cast(x > 0., dtype)

    return y, grad

def modify_backprop(model):
    for layer in model.layers:
        if hasattr(layer, 'activation') and layer.activation == tf.keras.activations.relu:
            layer.activation = custom_relu

    return model

def interpolate(x, y, xnew):
    f = spinterp.interp1d(x, y, kind=2)
    ynew = f(xnew)
    return ynew

def grad_cam_fn(input_model, swings, category_index, layer_name):
    modified_outputs = [input_model.get_layer(layer_name).output, input_model.output]
    modified_model = tf.keras.Model(inputs=input_model.input, outputs=modified_outputs)

    inp = tf.keras.layers.Input(shape=SHAPE)
    conv_outputs, preds = modified_model(inp)
    outputs = target_category_loss(preds, category_index, nb_classes)
    model = tf.keras.Model(inputs=inp, outputs=[outputs, conv_outputs])

    with tf.GradientTape() as tape:
        out, conv_out = model(swings, training=False)
        loss = tf.reduce_sum(out)

    grads = tape.gradient(loss, conv_out)
    grads = normalize(grads)

    conv_out, grads_val = conv_out[0].numpy(), grads[0].numpy()

    weights = np.mean(grads_val, axis=0)
    cam = np.ones(conv_out.shape[0:1], dtype=np.float32)

    for i, w in enumerate(weights):
        cam += w * conv_out[:, i]

    # 2021.08.25: ReLU fixed
    cam = np.maximum(cam, 0)
    heatmap = cam / (np.max(cam) + 1e-5)

    return cam, heatmap

def saliency_function(swings, model, activation_layer):
    layer_output = model.get_layer(activation_layer).output
    max_output = tf.reduce_max(layer_output, axis=2)
    model = tf.keras.Model(inputs=model.input, outputs=max_output)

    swings = tf.constant(swings)
    with tf.GradientTape() as tape:
        tape.watch(swings)
        out = tf.reduce_sum(model(swings, training=False))

    saliency = tape.gradient(out, swings)
    return saliency

x_test = np.load('../dataset/normalized/X_test_norm.npz')['arr_0']
y_test = np.load('../dataset/normalized/y_test.npz')['arr_0']
model = tf.keras.models.load_model(
    'saved_model/convnet_{}_{}.h5'.format(task[0], task[1]))

x_test = x_test[..., sample[0]:sample[1], sensor]

# ->> Infer all cases and save as .npz
for i in range(x_test.shape[0]):
    # ->> prediction
    x, y = x_test[i][np.newaxis], y_test[i] # x: [1, length, sensors], y: scalar
    predictions = model.predict(x) # predictions: [1, length]
    top_1 = predictions[0].argmax() # top_1: scalar
    print('Predicted class: {}, Actual class: {}'.format(top_1, y))

    # ->> Gradient-weighted Class Activation Map
    predicted_class = predictions[0].argmax() # predictions: scalar
    cam, heatmap = grad_cam_fn(model, x, predicted_class, 'block5_conv4')

    # ->> interpolation for heat map 
    heatmap_length = heatmap.shape[0]
    heatmap_coord = list(range(heatmap_length))
    swing_coord = np.linspace(0, heatmap_length - 1, swing_length)
    heatmap_ext = interpolate(heatmap_coord, heatmap, swing_coord)

    # ->> Guided Grad-CAM
    guided_model = modify_backprop(model)
    saliencies = saliency_function(x, guided_model, activation_layer='block5_conv4')
    guided_gradcam = saliencies[0].numpy() * heatmap_ext[..., np.newaxis]

    np.savez(
        os.path.join(save_root, '{}.npz'.format(i)), 
        predicted_class=predicted_class, 
        actual_class=y, 
        heatmap=heatmap,
        swing=x[0], 
        heatmap_ext=heatmap_ext, 
        guided_gradcam=guided_gradcam, 
        saliency=saliencies[0])
