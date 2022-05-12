import os
import numpy as np
import matplotlib.pyplot as plt

# ->> load matplotlib style
plt.style.use('golf.mplstyle')

input_root = '../multi-sensor/' # input root
output_root = 'multi-sensor-fig_v3/' # output root
if not os.path.exists(output_root):
    os.makedirs(output_root)

# ->> explore effects on multi-sensor usages
sensors = [
    'Gyro', 
    'Acc', 
    'SG', 
    'All', 
]
n_classes = 19
gap_length = 5

# ->> figs and axes
figs, axes = list(), list()
for i in range(n_classes): # 19 categories
    fig = plt.figure(figsize=(8, 4.5))
    ax = fig.add_subplot()
    figs.append(fig)
    axes.append(ax)

heatmaps = [[] for _ in range(n_classes)]

for sensor in sensors:
    input_path = os.path.join(input_root, sensor)
    
    names = os.listdir(input_path)
    for name in names:
        fullname = os.path.join(input_path, name)
        npz = np.load(fullname)

        predicted_class, actual_class = npz['predicted_class'], npz['actual_class']

        if predicted_class == actual_class: # correctly classified
            heatmap_ext = npz['heatmap_ext']
            if sensor == 'Gyro' and predicted_class == 2:
                print(heatmap_ext == None)
            heatmaps[int(actual_class)] += [heatmap_ext]

    for i in range(len(heatmaps)):
        if heatmaps[i] == []:
            for _ in range(10):
                heatmaps[i] += [np.zeros((645, ))]

    gap = np.ndarray((645, ))
    gap.fill(np.inf)
    for heatmap in heatmaps:
        for _ in range(gap_length):
            heatmap += [gap]

for i in range(n_classes):
    ax, fig = axes[i], figs[i]
    title = 'Predicted class: {}, Actual class: {}'.format(i, i)
    ax.set_title(title)
    # try:
    ax.imshow(heatmaps[i])
    # except Exception as e:
    #     print(e)
    # finally:
    ax.set_xlabel('Time [ms]')
    ax.set_ylabel('Swing No.\nSU Gyro. / Acc. / SG / All')
    ax.set_xlim(0, 645)
    fig.tight_layout()

    fig.savefig(os.path.join(output_root, 'label-{}.jpg'.format(i)))