import os
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('golf.mplstyle') # load matplotlib style

input_path = 'multi-sensor/' # input path
output_path = 'multi-sensor-fig/' # output path

if not os.path.exists(output_path):
    os.makedirs(output_path)

# ->> explore effects on multi-sensor
sensors = [
    'All', 
    'SG', 
    'Acc', 
    'Gyro', 
] 

def load_npz(path, name):
    fullname = os.path.join(path, name)
    npz = np.load(fullname)
    swing, heatmap_ext, guided_gradcam = npz['swing'], npz['heatmap_ext'], npz['guided_gradcam']
    predicted_class, actual_class = npz['predicted_class'], npz['actual_class']

    return swing, heatmap_ext, guided_gradcam, predicted_class, actual_class

def plot_fn_all(swing, heatmap_ext, guided_gradcam, title, fullname_prefix):
    figsize = (8, 4.5)
    # ->> plot sg
    plt.clf() 
    fig, ax = plt.subplots(figsize=figsize)   
    ax.plot(swing[:, 0], label='SG1')
    ax.plot(swing[:, 1], label='SG2')
    ax.plot(heatmap_ext, linestyle='--', color='orangered', label='Heatmap')
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
    ax.plot(heatmap_ext, linestyle='--', color='orangered', label='Heatmap')
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
    ax.plot(heatmap_ext, linestyle='--', color='orangered', label='Heatmap')
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
    ax.plot(heatmap_ext, linestyle='--', color='orangered', label='Heatmap')
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
    ax.plot(heatmap_ext, linestyle='--', color='orangered', label='Heatmap')
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
    ax.plot(heatmap_ext, linestyle='--', color='orangered', label='Heatmap')
    ax.set_title(title)
    ax.set_xlabel('Time [ms]')
    ax.set_ylabel(r'Gyro. guided grad-cam')
    ax.set_xlim(0, 645)
    ax.legend()
    plt.savefig(fullname_prefix + '_gyro_ggcam.jpg')
    plt.close()

def plot_fn_sg(swing, heatmap_ext, guided_gradcam, title, fullname_prefix):
    figsize = (8, 4.5)
    # ->> plot sg
    plt.clf() 
    fig, ax = plt.subplots(figsize=figsize)   
    ax.plot(swing[:, 0], label='SG1')
    ax.plot(swing[:, 1], label='SG2')
    ax.plot(heatmap_ext, linestyle='--', color='orangered', label='Heatmap')
    ax.set_title(title)
    ax.set_xlabel('Time [ms]')
    ax.set_ylabel(r'Strain [m$\epsilon$]')
    ax.set_xlim(0, 645)
    ax.legend()
    plt.savefig(fullname_prefix + '_sg.jpg')
    plt.close()

    # ->> plot guided grad-cam of sg
    plt.clf() 
    fig, ax = plt.subplots(figsize=figsize)   
    ax.plot(guided_gradcam[:, 0], label='SG1')
    ax.plot(guided_gradcam[:, 1], label='SG2')
    ax.plot(heatmap_ext, linestyle='--', color='orangered', label='Heatmap')
    ax.set_title(title)
    ax.set_xlabel('Time [ms]')
    ax.set_ylabel(r'SG guided grad-cam')
    ax.set_xlim(0, 645)
    ax.legend()
    plt.savefig(fullname_prefix + '_sg_ggcam.jpg')
    plt.close()

def plot_fn_acc(swing, heatmap_ext, guided_gradcam, title, fullname_prefix):
    figsize = (8, 4.5)

    # ->> plot acc.
    plt.clf() 
    fig, ax = plt.subplots(figsize=figsize)   
    ax.plot(swing[:, 0], label='AccX')
    ax.plot(swing[:, 1], label='AccY')
    ax.plot(swing[:, 2], label='AccZ')
    ax.plot(heatmap_ext, linestyle='--', color='orangered', label='Heatmap')
    ax.set_title(title)
    ax.set_xlabel('Time [ms]')
    ax.set_ylabel(r'Acceleration [m/s$^{2}$]')
    ax.set_xlim(0, 645)
    ax.legend()
    plt.savefig(fullname_prefix + '_acc.jpg')
    plt.close()

    # ->> plot guided grad-cam of acc.
    plt.clf()
    fig, ax = plt.subplots(figsize=figsize)   
    ax.plot(guided_gradcam[:, 0], label='AccX')
    ax.plot(guided_gradcam[:, 1], label='AccY')
    ax.plot(guided_gradcam[:, 2], label='AccZ')
    ax.plot(heatmap_ext, linestyle='--', color='orangered', label='Heatmap')
    ax.set_title(title)
    ax.set_xlabel('Time [ms]')
    ax.set_ylabel(r'Acc. guided grad-cam')
    ax.set_xlim(0, 645)
    ax.legend()
    plt.savefig(fullname_prefix + '_acc_ggcam.jpg')
    plt.close()

def plot_fn_gyro(swing, heatmap_ext, guided_gradcam, title, fullname_prefix):
    figsize = (8, 4.5)
    # ->> plot gyro.
    plt.clf() 
    fig, ax = plt.subplots(figsize=figsize)   
    ax.plot(swing[:, 0], label='GyroX')
    ax.plot(swing[:, 1], label='GyroY')
    ax.plot(swing[:, 2], label='GyroZ')
    ax.plot(heatmap_ext, linestyle='--', color='orangered', label='Heatmap')
    ax.set_title(title)
    ax.set_xlabel('Time [ms]')
    ax.set_ylabel(r'Angular speed [deg/s]')
    ax.set_xlim(0, 645)
    ax.legend()
    plt.savefig(fullname_prefix + '_gyro.jpg')
    plt.close()

    # ->> plot guided grad-cam of gyro.
    plt.clf()
    fig, ax = plt.subplots(figsize=figsize)   
    ax.plot(guided_gradcam[:, 0], label='GyroX')
    ax.plot(guided_gradcam[:, 1], label='GyroY')
    ax.plot(guided_gradcam[:, 2], label='GyroZ')
    ax.plot(heatmap_ext, linestyle='--', color='orangered', label='Heatmap')
    ax.set_title(title)
    ax.set_xlabel('Time [ms]')
    ax.set_ylabel(r'Gyro. guided grad-cam')
    ax.set_xlim(0, 645)
    ax.legend()
    plt.savefig(fullname_prefix + '_gyro_ggcam.jpg')
    plt.close()

def load_and_plot(path, name, save_fullname_prefix, sensor):
    # ->> load swing
    swing, heatmap_ext, guided_gradcam, predicted_class, actual_class = load_npz(path, name)
    title = 'Predicted class: {}, Actual class: {}'.format(predicted_class, actual_class)
    if sensor is 'All':
        plot_fn_all(swing, heatmap_ext, guided_gradcam, title, save_fullname_prefix)
    elif sensor is 'SG':
        plot_fn_sg(swing, heatmap_ext, guided_gradcam, title, save_fullname_prefix)
    elif sensor is 'Acc':
        plot_fn_acc(swing, heatmap_ext, guided_gradcam, title, save_fullname_prefix)
    elif sensor is 'Gyro':
        plot_fn_gyro(swing, heatmap_ext, guided_gradcam, title, save_fullname_prefix)

for sensor in sensors:
    if not os.path.exists(os.path.join(output_path, sensor)):
        os.makedirs(os.path.join(output_path, sensor))
    
    load_and_plot(os.path.join(input_path, sensor), '0.npz', os.path.join(output_path, sensor, '0'), sensor)