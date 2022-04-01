import os
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('golf.mplstyle') # load matplotlib style

input_path = '../multi-time/' # input path
output_path = 'multi-time-fig/' # output path

if not os.path.exists(output_path):
    os.makedirs(output_path)

# ->> explore effects on multi-time 
times = [
    '645',
    '600', 
    '550', 
    '500', 
    '450', 
]

def load_npz(path, name):
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
    ax.set_xlim(0, 645)
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
    ax.set_xlim(0, 645)
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
    ax.set_xlim(0, 645)
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
    ax.set_xlim(0, 645)
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
    ax.set_xlim(0, 645)
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
    ax.set_xlim(0, 645)
    ax.legend()
    plt.savefig(fullname_prefix + '_gyro_ggcam.jpg')
    plt.close()

def load_and_plot(path, name, save_fullname_prefix):
    # ->> load swing
    swing, heatmap_ext, guided_gradcam, predicted_class, actual_class = load_npz(path, name)
    title = 'Predicted class: {}, Actual class: {}'.format(predicted_class, actual_class)
    plot_fn(swing, heatmap_ext, guided_gradcam, title, save_fullname_prefix)

for time in times:
    if not os.path.exists(os.path.join(output_path, time)):
        os.makedirs(os.path.join(output_path, time))
    
    load_and_plot(os.path.join(input_path, time), '0.npz', os.path.join(output_path, time, '0'))