import os
import numpy as np
import matplotlib.pyplot as plt

width_spectral = 10

# ->> load matplotlib style
plt.style.use('golf.mplstyle')

# ->> data root and save root
data_path = '../cam/'
out_path = 'attn-consistency/'
if not os.path.exists(out_path):
    os.makedirs(out_path)

# ->> figs and axes
figs, axes = [], []
for i in range(19): # 19 categories
    fig = plt.figure(figsize=(8, 4.5))
    ax = fig.add_subplot()
    figs.append(fig)
    axes.append(ax)

heatmaps = [[] for _ in range(19)]

names = os.listdir(data_path)
for name in names:
    fullname = os.path.join(data_path, name)
    npz = np.load(fullname)

    predicted_class, actual_class = npz['predicted_class'], npz['actual_class']

    if predicted_class == actual_class: # correctly classified
        heatmap_ext = npz['heatmap_ext']
        heatmaps[int(actual_class)] += [heatmap_ext]

for i in range(19):
    ax, fig = axes[i], figs[i]
    title = 'Predicted class: {}, Actual class: {}'.format(i, i)
    ax.set_title(title)
    ax.imshow(heatmaps[i], cmap='rainbow', aspect='auto', interpolation='nearest', origin='lower')
    ax.set_xlabel('Time [ms]')
    ax.set_ylabel('Swing No.')
    fig.tight_layout()

    fig.savefig(os.path.join(out_path, 'label-{}.jpg'.format(i)))

