import os
import numpy as np
import matplotlib.pyplot as plt

# ->> load matplotlib style
plt.style.use('golf.mplstyle')

input_root = '../multi-time/' # input path
output_root = 'multi-time-fig_v2/' # output path

# ->> explore effects on multi-time selections
times = [
    '645',
    '600', 
    '550', 
    '500', 
    '450', 
]
n_classes = 19

for time in times:
    input_path = os.path.join(input_root, time)
    output_path = os.path.join(output_root, time)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    # ->> figs and axes
    figs, axes = list(), list()
    for i in range(n_classes): # 19 categories
        fig = plt.figure(figsize=(8, 4.5))
        ax = fig.add_subplot()
        figs.append(fig)
        axes.append(ax)

    heatmaps = [[] for _ in range(n_classes)]

    names = os.listdir(input_path)
    for name in names:
        fullname = os.path.join(input_path, name)
        npz = np.load(fullname)

        predicted_class, actual_class = npz['predicted_class'], npz['actual_class']

        if predicted_class == actual_class: # correctly classified
            heatmap_ext = npz['heatmap_ext']
            heatmaps[int(actual_class)] += [heatmap_ext]

    for i in range(n_classes):
        ax, fig = axes[i], figs[i]
        title = 'Predicted class: {}, Actual class: {}'.format(i, i)
        ax.set_title(title)
        ax.imshow(heatmaps[i], cmap='rainbow', aspect='auto', interpolation='nearest', origin='lower')
        ax.set_xlabel('Time [ms]')
        ax.set_ylabel('Swing No.')
        ax.set_xlim(0, 645)
        fig.tight_layout()

        fig.savefig(os.path.join(output_path, 'label-{}.jpg'.format(i)))
        