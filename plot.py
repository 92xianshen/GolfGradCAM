import numpy as np
import matplotlib.pyplot as plt

# ->> load matplotlib style
plt.style.use('golf.mplstyle')

# ->> plot function
def plot_fn(swing, heatmap, ggcam, title):
    # plt.clf()
    # 3 sensors + 1 heatmap + 3 gradient backpropagation
    fig, axs = plt.subplots(3, 2, figsize=(16, 9))
    ax_sg, ax_sgggcam, ax_acc, ax_accggcam, ax_gyro, ax_gyroggcam = axs.flatten()

    # sg
    ax_sg.plot(swing[:, 0], label='SG1')
    ax_sg.plot(swing[:, 1], label='SG2')
    ax_sg.plot(heatmap, linestyle='--', color='r', label='Heatmap')
    ax_sg.set_title(title)
    ax_sg.set_xlabel('Time [ms]')
    ax_sg.set_ylabel(r'Strain [m$\epsilon$]')
    ax_sg.legend()

    # acc.
    ax_acc.plot(swing[:, 2], label='AccX')
    ax_acc.plot(swing[:, 3], label='AccY')
    ax_acc.plot(swing[:, 4], label='AccZ')
    ax_acc.plot(heatmap, linestyle='--', color='r', label='Heatmap')
    ax_acc.set_xlabel('Time [ms]')
    ax_acc.set_ylabel(r'Acceleration [m/s$^{2}$]')
    ax_acc.legend()

    # gyro.
    ax_gyro.plot(swing[:, 5], label='GyroX')
    ax_gyro.plot(swing[:, 6], label='GyroY')
    ax_gyro.plot(swing[:, 7], label='GyroZ')
    ax_gyro.plot(heatmap, linestyle='--', color='r', label='Heatmap')
    ax_gyro.set_xlabel('Time [ms]')
    ax_gyro.set_ylabel(r'Angular speed [deg/s]')
    ax_gyro.legend()

    # sg guided grad-cam
    ax_sgggcam.plot(ggcam[:, 0], label='SG1')
    ax_sgggcam.plot(ggcam[:, 1], label='SG2')
    ax_sgggcam.plot(heatmap, linestyle='--', color='r', label='Heatmap')
    ax_sgggcam.set_xlabel('Time [ms]')
    ax_sgggcam.set_ylabel(r'SG guided grad-cam')
    ax_sgggcam.legend()

    # acc. guided grad-cam
    ax_accggcam.plot(ggcam[:, 2], label='AccX')
    ax_accggcam.plot(ggcam[:, 3], label='AccY')
    ax_accggcam.plot(ggcam[:, 4], label='AccZ')
    ax_accggcam.plot(heatmap, linestyle='--', color='r', label='Heatmap')
    ax_accggcam.set_xlabel('Time [ms]')
    ax_accggcam.set_ylabel(r'Acc. guided grad-cam')
    ax_accggcam.legend()

    # gyro. guided grad-cam
    ax_gyroggcam.plot(ggcam[:, 5], label='GyroX')
    ax_gyroggcam.plot(ggcam[:, 6], label='GyroY')
    ax_gyroggcam.plot(ggcam[:, 7], label='GyroZ')
    ax_gyroggcam.plot(heatmap, linestyle='--', color='r', label='Heatmap')
    ax_gyroggcam.set_xlabel('Time [ms]')
    ax_gyroggcam.set_ylabel(r'Gyro. guided grad-cam')
    ax_gyroggcam.legend()

    plt.show()

# ->> plot function
def save_fn(swing, heatmap, ggcam, title, fullname):
    plt.clf()
    # 3 sensors + 1 heatmap + 3 gradient backpropagation
    fig, axs = plt.subplots(3, 2, figsize=(16, 9))
    ax_sg, ax_sgggcam, ax_acc, ax_accggcam, ax_gyro, ax_gyroggcam = axs.flatten()

    # sg
    ax_sg.plot(swing[:, 0], label='SG1')
    ax_sg.plot(swing[:, 1], label='SG2')
    ax_sg.plot(heatmap, linestyle='--', color='r', label='Heatmap')
    ax_sg.set_title(title)
    ax_sg.set_xlabel('Time [ms]')
    ax_sg.set_ylabel(r'Strain [m$\epsilon$]')
    ax_sg.legend()

    # acc.
    ax_acc.plot(swing[:, 2], label='AccX')
    ax_acc.plot(swing[:, 3], label='AccY')
    ax_acc.plot(swing[:, 4], label='AccZ')
    ax_acc.plot(heatmap, linestyle='--', color='r', label='Heatmap')
    ax_acc.set_xlabel('Time [ms]')
    ax_acc.set_ylabel(r'Acceleration [m/s$^{2}$]')
    ax_acc.legend()

    # gyro.
    ax_gyro.plot(swing[:, 5], label='GyroX')
    ax_gyro.plot(swing[:, 6], label='GyroY')
    ax_gyro.plot(swing[:, 7], label='GyroZ')
    ax_gyro.plot(heatmap, linestyle='--', color='r', label='Heatmap')
    ax_gyro.set_xlabel('Time [ms]')
    ax_gyro.set_ylabel(r'Angular speed [deg/s]')
    ax_gyro.legend()

    # sg guided grad-cam
    ax_sgggcam.plot(ggcam[:, 0], label='SG1')
    ax_sgggcam.plot(ggcam[:, 1], label='SG2')
    ax_sgggcam.plot(heatmap, linestyle='--', color='r', label='Heatmap')
    ax_sgggcam.set_xlabel('Time [ms]')
    ax_sgggcam.set_ylabel(r'SG guided grad-cam')
    ax_sgggcam.legend()

    # acc. guided grad-cam
    ax_accggcam.plot(ggcam[:, 2], label='AccX')
    ax_accggcam.plot(ggcam[:, 3], label='AccY')
    ax_accggcam.plot(ggcam[:, 4], label='AccZ')
    ax_accggcam.plot(heatmap, linestyle='--', color='r', label='Heatmap')
    ax_accggcam.set_xlabel('Time [ms]')
    ax_accggcam.set_ylabel(r'Acc. guided grad-cam')
    ax_accggcam.legend()

    # gyro. guided grad-cam
    ax_gyroggcam.plot(ggcam[:, 5], label='GyroX')
    ax_gyroggcam.plot(ggcam[:, 6], label='GyroY')
    ax_gyroggcam.plot(ggcam[:, 7], label='GyroZ')
    ax_gyroggcam.plot(heatmap, linestyle='--', color='r', label='Heatmap')
    ax_gyroggcam.set_xlabel('Time [ms]')
    ax_gyroggcam.set_ylabel(r'Gyro. guided grad-cam')
    ax_gyroggcam.legend()

    plt.savefig(fullname)
    plt.close()

npz = np.load('cam/0.npz')
swing, heatmap_ext, guided_gradcam, predicted_class, actual_class = npz['swing'], npz['heatmap_ext'], npz['guided_gradcam'], npz['predicted_class'], npz['actual_class']

title = 'Predicted class: {}, Actual class: {}'.format(predicted_class, actual_class)

plot_fn(swing, heatmap_ext, guided_gradcam, title)
save_fn(swing, heatmap_ext, guided_gradcam, title, '0.jpg')

# plot_fn(x[0], heatmap_ext, saliency[0], title)
# show_fn(x[0], heatmap_ext, title, os.path.join(save_root, '{}.jpg'.format(i)))