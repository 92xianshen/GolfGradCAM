import os
import numpy as np
import matplotlib.pyplot as plt

# ->> load matplotlib style
plt.style.use('golf.mplstyle')
figsize = (8, 4.5)

# ->> data root and save root
data_path = 'cam'
save_path = 'attn-consistency'

# ->> figs and axes
# figs, axes = [], [] # axis 0 denotes labels, axis 1 denotes sensors
# for i in range(19): # 19 categories
#     figs0, axes0 = [], []
#     for j in range(6): # sg, acc., gyro., sgggcam, accggcam, gyroggcam
#         fig, ax = plt.subplots(figsize=figsize)
#         figs0.append(fig)
#         axes0.append(ax)

#     figs.append(figs0)
#     axes.append(axes0)
figs, axes = [], []
for i in range(19): # 19 categories
    fig, ax = plt.subplots(figsize=figsize)
    figs.append(fig)
    axes.append(ax)

names = os.listdir(data_path)
for name in names:
    fullname = os.path.join(data_path, name)
    npz = np.load(fullname)

    predicted_class, actual_class = npz['predicted_class'], npz['actual_class']

    if predicted_class == actual_class: # correctly classified
        heatmap_ext = npz['heatmap_ext']
        ax = axes[int(actual_class)]
        ax.plot(heatmap_ext, color='orangered')

for i in range(19):
    ax = axes[i]
    title = 'Predicted class: {}, Actual class: {}'.format(i, i)
    ax.set_title(title)
    ax.set_xlabel('Time [ms]')
    ax.set_ylabel('Heatmap')
    figs[i].savefig(os.path.join(save_path, 'label-{}.jpg'.format(i)))




        