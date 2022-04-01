import os
import numpy as np
import matplotlib.pyplot as plt

# ->> load matplotlib style
plt.style.use('golf.mplstyle')

path = '../cam/'
save_path = 'misclassified'

correct_swings, incorrect_swings = dict(), dict()

def load_npz(name):
    fullname = os.path.join(path, name)
    npz = np.load(fullname)
    swing, heatmap_ext, guided_gradcam = npz['swing'], npz['heatmap_ext'], npz['guided_gradcam']
    predicted_class, actual_class = npz['predicted_class'], npz['actual_class']

    return swing, heatmap_ext, guided_gradcam, predicted_class, actual_class

def plot_fn(swing, heatmap_ext, guided_gradcam, title, fullname_prefix):
    figsize = (8, 4.5)
    # ->> plot sg
    plt.clf() 
    fig, ax = plt.subplots(figsize=figsize)   
    ax.plot(swing[:, 0], label='SG1')
    ax.plot(swing[:, 1], label='SG2')
    ax.imshow(heatmap_ext.reshape(1, -1), alpha=.75, extent=(0, swing.shape[0], -.05, 1.05))
    ax.set_title(title)
    ax.set_xlabel('Time [ms]')
    ax.set_ylabel(r'Strain [m$\epsilon$]')
    ax.legend()
    plt.savefig(fullname_prefix + '_sg.jpg')
    plt.close()

    # ->> plot acc.
    plt.clf() 
    fig, ax = plt.subplots(figsize=figsize)   
    ax.plot(swing[:, 2], label='AccX')
    ax.plot(swing[:, 3], label='AccY')
    ax.plot(swing[:, 4], label='AccZ')
    ax.imshow(heatmap_ext.reshape(1, -1), alpha=.75, extent=(0, swing.shape[0], -.05, 1.05))
    ax.set_title(title)
    ax.set_xlabel('Time [ms]')
    ax.set_ylabel(r'Acceleration [m/s$^{2}$]')
    ax.legend()
    plt.savefig(fullname_prefix + '_acc.jpg')
    plt.close()

    # ->> plot gyro.
    plt.clf() 
    fig, ax = plt.subplots(figsize=figsize)   
    ax.plot(swing[:, 5], label='GyroX')
    ax.plot(swing[:, 6], label='GyroY')
    ax.plot(swing[:, 7], label='GyroZ')
    ax.imshow(heatmap_ext.reshape(1, -1), alpha=.75, extent=(0, swing.shape[0], -.05, 1.05))
    ax.set_title(title)
    ax.set_xlabel('Time [ms]')
    ax.set_ylabel(r'Angular speed [deg/s]')
    ax.legend()
    plt.savefig(fullname_prefix + '_gyro.jpg')
    plt.close()

    # ->> plot guided grad-cam of sg
    plt.clf() 
    fig, ax = plt.subplots(figsize=figsize)   
    ax.plot(guided_gradcam[:, 0], label='SG1')
    ax.plot(guided_gradcam[:, 1], label='SG2')
    bottom, top = guided_gradcam[:, 0:2].min(), guided_gradcam[:, 0:2].max()
    padding = (top - bottom) / 20.
    ax.imshow(heatmap_ext.reshape(1, -1), alpha=.75, extent=(0, guided_gradcam.shape[0], bottom - padding, top + padding))
    ax.set_title(title)
    ax.set_xlabel('Time [ms]')
    ax.set_ylabel(r'SG guided grad-cam')
    ax.legend()
    plt.savefig(fullname_prefix + '_sg_ggcam.jpg')
    plt.close()

    # ->> plot guided grad-cam of acc.
    plt.clf()
    fig, ax = plt.subplots(figsize=figsize)   
    ax.plot(guided_gradcam[:, 2], label='AccX')
    ax.plot(guided_gradcam[:, 3], label='AccY')
    ax.plot(guided_gradcam[:, 4], label='AccZ')
    bottom, top = guided_gradcam[:, 2:5].min(), guided_gradcam[:, 2:5].max()
    padding = (top - bottom) / 20.
    ax.imshow(heatmap_ext.reshape(1, -1), alpha=.75, extent=(0, guided_gradcam.shape[0], bottom - padding, top + padding))
    ax.set_title(title)
    ax.set_xlabel('Time [ms]')
    ax.set_ylabel(r'Acc. guided grad-cam')
    ax.legend()
    plt.savefig(fullname_prefix + '_acc_ggcam.jpg')
    plt.close()

    # ->> plot guided grad-cam of gyro.
    plt.clf()
    fig, ax = plt.subplots(figsize=figsize)   
    ax.plot(guided_gradcam[:, 5], label='GyroX')
    ax.plot(guided_gradcam[:, 6], label='GyroY')
    ax.plot(guided_gradcam[:, 7], label='GyroZ')
    bottom, top = guided_gradcam[:, 5:8].min(), guided_gradcam[:, 5:8].max()
    padding = (top - bottom) / 20.
    ax.imshow(heatmap_ext.reshape(1, -1), alpha=.75, extent=(0, guided_gradcam.shape[0], bottom - padding, top + padding))
    ax.set_title(title)
    ax.set_xlabel('Time [ms]')
    ax.set_ylabel(r'Gyro. guided grad-cam')
    ax.legend()
    plt.savefig(fullname_prefix + '_gyro_ggcam.jpg')
    plt.close()

def load_and_plot(name, save_fullname_prefix):
    # ->> load swing
    swing, heatmap_ext, guided_gradcam, predicted_class, actual_class = load_npz(name)
    title = 'Predicted class: {}, Actual class: {}'.format(predicted_class, actual_class)
    plot_fn(swing, heatmap_ext, guided_gradcam, title, save_fullname_prefix)

names = os.listdir(path)

# get all misclassified swings
for name in names:
    fullname = os.path.join(path, name)
    npz = np.load(fullname)

    predicted_class, actual_class = npz['predicted_class'], npz['actual_class']

    if predicted_class == actual_class:
        if str(int(actual_class)) not in correct_swings.keys():
            correct_swings[str(int(actual_class))] = []
        correct_swings[str(int(actual_class))] += [name]
    else:
        incorrect_swings[name] = {'predicted_class': int(predicted_class), 'actual_class': int(actual_class)}

print(incorrect_swings)

# heatmap of incorrect swings, heatmap of predicted class, and heatmap of actual class
for name in incorrect_swings.keys():
    misclassified_swing_idx = os.path.splitext(name)[0]

    # ->> create save folder
    if not os.path.exists(os.path.join(save_path, misclassified_swing_idx)):
        os.makedirs(os.path.join(save_path, misclassified_swing_idx))

    # ->> load predicted_class, actual_class
    predicted_class = incorrect_swings[name]['predicted_class']
    actual_class = incorrect_swings[name]['actual_class']

    # # ->> load and plot incorrect swing
    save_fullname_prefix = os.path.join(save_path, misclassified_swing_idx, misclassified_swing_idx)
    load_and_plot(name, save_fullname_prefix)

    # ->> load and plot predicted class
    name_predicted = correct_swings[str(predicted_class)][0] # select the first swing as reference
    save_fullname_prefix_predicted = os.path.join(save_path, misclassified_swing_idx, os.path.splitext(name_predicted)[0])
    load_and_plot(name_predicted, save_fullname_prefix_predicted)

    # ->> load and plot actual class
    name_actual = correct_swings[str(actual_class)][0] # select the first swing as reference
    save_fullname_prefix_actual = os.path.join(save_path, misclassified_swing_idx, os.path.splitext(name_actual)[0])
    load_and_plot(name_actual, save_fullname_prefix_actual)