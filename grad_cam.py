import os
import numpy as np
import tensorflow as tf
from scipy import interpolate as spinterp
import matplotlib.pyplot as plt

task = ('All', '645')

swing_no = 2
nb_classes = 19
SHAPE = [645, 8]
swing_length = 645

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

# Save folder
save_root = 'cam/'
if not os.path.exists(save_root):
    os.makedirs(save_root)

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

def grad_cam(input_model, swing, category_index, layer_name):
    modified_outputs = [input_model.get_layer(layer_name).output, input_model.output]
    modified_model = tf.keras.Model(inputs=input_model.input, outputs=modified_outputs)

    inp = tf.keras.layers.Input(shape=SHAPE)
    conv_outputs, preds = modified_model(inp)
    outputs = target_category_loss(preds, category_index, nb_classes)
    model = tf.keras.Model(inputs=inp, outputs=[outputs, conv_outputs])

    with tf.GradientTape() as tape:
        out, conv_out = model(swing, training=False)
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

def saliency_function(swing, model, activation_layer):
    layer_output = model.get_layer(activation_layer).output
    max_output = tf.reduce_max(layer_output, axis=2)
    model = tf.keras.Model(inputs=model.input, outputs=max_output)

    swing = tf.constant(swing)
    with tf.GradientTape() as tape:
        tape.watch(swing)
        out = tf.reduce_sum(model(swing, training=False))

    saliency = tape.gradient(out, swing)
    return saliency

def show_fn(swing, heatmap, title, fullname):
    plt.clf()
    fig, axs = plt.subplots(4, 1)
    ax_sg, ax_acc, ax_gyro, ax_heatmap = axs.flatten()

    ax_sg.spines['right'].set_visible(False)
    ax_sg.spines['top'].set_visible(False)
    ax_sg.plot(swing[:, 0])
    ax_sg.plot(swing[:, 1])
    ax_sg.set_title(title)

    ax_acc.spines['right'].set_visible(False)
    ax_acc.spines['top'].set_visible(False)
    ax_acc.plot(swing[:, 2])
    ax_acc.plot(swing[:, 3])
    ax_acc.plot(swing[:, 4])

    ax_gyro.spines['right'].set_visible(False)
    ax_gyro.spines['top'].set_visible(False)
    ax_gyro.plot(swing[:, 5])
    ax_gyro.plot(swing[:, 6])
    ax_gyro.plot(swing[:, 7])

    ax_heatmap.spines['right'].set_visible(False)
    ax_heatmap.spines['top'].set_visible(False)
    ax_heatmap.plot(heatmap)

    plt.savefig(fullname)
    plt.close()

x_test = np.load('../dataset/normalized/X_test_norm.npz')['arr_0']
y_test = np.load('../dataset/normalized/y_test.npz')['arr_0']
model = tf.keras.models.load_model(
    'saved_model/convnet_{}_{}.h5'.format(task[0], task[1]))

x_test = x_test[..., sample[0]:sample[1], sensor]

# ->> Infer all cases and save as .npz
for i in range(x_test.shape[0]):
    x, y = x_test[i][np.newaxis], y_test[i]
    predictions = model.predict(x)
    top_1 = predictions[0].argmax()
    print('Predicted class: {}, Actual class: {}'.format(top_1, y))

    predicted_class = predictions[0].argmax()
    cam, heatmap = grad_cam(model, x, predicted_class, 'block5_conv4')

    heatmap_length = heatmap.shape[0]
    heatmap_coord = list(range(heatmap_length))
    swing_coord = np.linspace(0, heatmap_length - 1, swing_length)
    heatmap_ext = interpolate(heatmap_coord, heatmap, swing_coord)

    title = 'Predicted class: {}, Actual class: {}'.format(top_1, y)
    show_fn(x[0], heatmap_ext, title, os.path.join(save_root, '{}.jpg'.format(i)))

    np.savez(
        os.path.join(save_root, '{}.npz'.format(i)), 
        predicted_class=predicted_class, 
        actual_class=y, 
        heatmap=heatmap,
        swing=x[0], 
        heatmap_ext=heatmap_ext)
