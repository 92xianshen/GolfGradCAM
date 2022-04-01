import os
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('golf.mplstyle') # load matplotlib style

save_path = 'multi-rf-fig/' # save path
data_path = '../multi-rf/' # data path

# ->> explore effect on multi-rfs (receptive fields, rf)
rfs = [
    'block1_conv2', # front layer
    'block2_conv2', 
    'block3_conv4', 
    'block4_conv4', 
    'block5_conv4', # back layer
]

def load_npz(name):
    fullname = os.path.join(data_path, name)
    npz = np.load(fullname)
    swing, heatmap_ext, guided_gradcam = npz['swing'], npz['heatmap_ext'], npz['guided_gradcam']
    predicted_class, actual_class = npz['predicted_class'], npz['actual_class']

    return swing, heatmap_ext, guided_gradcam, predicted_class, actual_class

# def plot_fn(swing, heatmap_ext, guided_gradcam, title, fullname_prefix):
#     figsize = (8, 4.5)
#     # ->> plot sg
#     plt.clf() 
#     fig, ax = plt.subplots(figsize=figsize)   
#     ax.plot(swing[:, 0], label='SG1')
#     ax.plot(swing[:, 1], label='SG2')
#     ax.imshow(heatmap_ext.reshape(1, -1), alpha=.75, extent=(0, swing.shape[0], -.05, 1.05))
#     ax.set_title(title)
#     ax.set_xlabel('Time [ms]')
#     ax.set_ylabel(r'Strain [m$\epsilon$]')
#     ax.legend()
#     plt.savefig(fullname_prefix + '_sg.jpg')
#     plt.close()

#     # ->> plot acc.
#     plt.clf() 
#     fig, ax = plt.subplots(figsize=figsize)   
#     ax.plot(swing[:, 2], label='AccX')
#     ax.plot(swing[:, 3], label='AccY')
#     ax.plot(swing[:, 4], label='AccZ')
#     ax.imshow(heatmap_ext.reshape(1, -1), alpha=.75, extent=(0, swing.shape[0], -.05, 1.05))
#     ax.set_title(title)
#     ax.set_xlabel('Time [ms]')
#     ax.set_ylabel(r'Acceleration [m/s$^{2}$]')
#     ax.legend()
#     plt.savefig(fullname_prefix + '_acc.jpg')
#     plt.close()

#     # ->> plot gyro.
#     plt.clf() 
#     fig, ax = plt.subplots(figsize=figsize)   
#     ax.plot(swing[:, 5], label='GyroX')
#     ax.plot(swing[:, 6], label='GyroY')
#     ax.plot(swing[:, 7], label='GyroZ')
#     ax.imshow(heatmap_ext.reshape(1, -1), alpha=.75, extent=(0, swing.shape[0], -.05, 1.05))
#     ax.set_title(title)
#     ax.set_xlabel('Time [ms]')
#     ax.set_ylabel(r'Angular speed [deg/s]')
#     ax.legend()
#     plt.savefig(fullname_prefix + '_gyro.jpg')
#     plt.close()

#     # ->> plot guided grad-cam of sg
#     plt.clf() 
#     fig, ax = plt.subplots(figsize=figsize)   
#     ax.plot(guided_gradcam[:, 0], label='SG1')
#     ax.plot(guided_gradcam[:, 1], label='SG2')
#     bottom, top = guided_gradcam[:, 0:2].min(), guided_gradcam[:, 0:2].max()
#     padding = (top - bottom) / 20.
#     ax.imshow(heatmap_ext.reshape(1, -1), alpha=.75, extent=(0, guided_gradcam.shape[0], bottom - padding, top + padding))
#     ax.set_title(title)
#     ax.set_xlabel('Time [ms]')
#     ax.set_ylabel(r'SG guided grad-cam')
#     ax.legend()
#     plt.savefig(fullname_prefix + '_sg_ggcam.jpg')
#     plt.close()

#     # ->> plot guided grad-cam of acc.
#     plt.clf()
#     fig, ax = plt.subplots(figsize=figsize)   
#     ax.plot(guided_gradcam[:, 2], label='AccX')
#     ax.plot(guided_gradcam[:, 3], label='AccY')
#     ax.plot(guided_gradcam[:, 4], label='AccZ')
#     bottom, top = guided_gradcam[:, 2:5].min(), guided_gradcam[:, 2:5].max()
#     padding = (top - bottom) / 20.
#     ax.imshow(heatmap_ext.reshape(1, -1), alpha=.75, extent=(0, guided_gradcam.shape[0], bottom - padding, top + padding))
#     ax.set_title(title)
#     ax.set_xlabel('Time [ms]')
#     ax.set_ylabel(r'Acc. guided grad-cam')
#     ax.legend()
#     plt.savefig(fullname_prefix + '_acc_ggcam.jpg')
#     plt.close()

#     # ->> plot guided grad-cam of gyro.
#     plt.clf()
#     fig, ax = plt.subplots(figsize=figsize)   
#     ax.plot(guided_gradcam[:, 5], label='GyroX')
#     ax.plot(guided_gradcam[:, 6], label='GyroY')
#     ax.plot(guided_gradcam[:, 7], label='GyroZ')
#     bottom, top = guided_gradcam[:, 5:8].min(), guided_gradcam[:, 5:8].max()
#     padding = (top - bottom) / 20.
#     ax.imshow(heatmap_ext.reshape(1, -1), alpha=.75, extent=(0, guided_gradcam.shape[0], bottom - padding, top + padding))
#     ax.set_title(title)
#     ax.set_xlabel('Time [ms]')
#     ax.set_ylabel(r'Gyro. guided grad-cam')
#     ax.legend()
#     plt.savefig(fullname_prefix + '_gyro_ggcam.jpg')
#     plt.close()

# ->> Plot all lines in one figure.
def plot_fn(swing, heatmap_ext, guided_gradcam, title, fullname_prefix):
    figsize = (8, 4.5)
    # ->> plot signals
    plt.clf() 
    fig, ax = plt.subplots(figsize=figsize)   
    ax.plot(swing[:, 0], label='SG1', linestyle='-')
    ax.plot(swing[:, 1], label='SG2', linestyle='-')
    ax.plot(swing[:, 2], label='AccX', linestyle='--')
    ax.plot(swing[:, 3], label='AccY', linestyle='--')
    ax.plot(swing[:, 4], label='AccZ', linestyle='--')
    ax.plot(swing[:, 5], label='GyroX', linestyle=':')
    ax.plot(swing[:, 6], label='GyroY', linestyle=':')
    ax.plot(swing[:, 7], label='GyroZ', linestyle=':')
    ax.imshow(heatmap_ext.reshape(1, -1), alpha=.75, extent=(0, swing.shape[0], -.05, 1.05))
    ax.set_title(title)
    ax.set_xlabel('Time [ms]')
    ax.set_ylabel(r'Strain/Acceleration/Angular speed [m$\epsilon$]/[m/s$^{2}$]/[deg/s]')
    ax.legend()
    plt.savefig(fullname_prefix + '.jpg')
    plt.close()

    # ->> plot guided grad-cams
    plt.clf() 
    fig, ax = plt.subplots(figsize=figsize)   
    ax.plot(guided_gradcam[:, 0], label='SG1', linestyle='-')
    ax.plot(guided_gradcam[:, 1], label='SG2', linestyle='-')
    ax.plot(guided_gradcam[:, 2], label='AccX', linestyle='--')
    ax.plot(guided_gradcam[:, 3], label='AccY', linestyle='--')
    ax.plot(guided_gradcam[:, 4], label='AccZ', linestyle='--')
    ax.plot(guided_gradcam[:, 5], label='GyroX', linestyle=':')
    ax.plot(guided_gradcam[:, 6], label='GyroY', linestyle=':')
    ax.plot(guided_gradcam[:, 7], label='GyroZ', linestyle=':')
    bottom, top = guided_gradcam[:, 0:8].min(), guided_gradcam[:, 0:8].max()
    padding = (top - bottom) / 20.
    ax.imshow(heatmap_ext.reshape(1, -1), alpha=.75, extent=(0, guided_gradcam.shape[0], bottom - padding, top + padding))
    ax.set_title(title)
    ax.set_xlabel('Time [ms]')
    ax.set_ylabel(r'SG/Acc./Gyro. guided grad-cam')
    ax.legend()
    plt.savefig(fullname_prefix + '_ggcam.jpg')
    plt.close()

def load_and_plot(name, save_fullname_prefix):
    # ->> load swing
    swing, heatmap_ext, guided_gradcam, predicted_class, actual_class = load_npz(name)
    title = 'Predicted class: {}, Actual class: {}'.format(predicted_class, actual_class)
    plot_fn(swing, heatmap_ext, guided_gradcam, title, save_fullname_prefix)

for rf in rfs:
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    load_and_plot(rf + '.npz', os.path.join(save_path, rf))

