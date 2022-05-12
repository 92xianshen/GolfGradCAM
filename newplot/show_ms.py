import os
import numpy as np

input_root = '../multi-sensor/' # input root
input_path = os.path.join(input_root, 'Gyro')

names = os.listdir(input_path)
for name in names:
    fullname = os.path.join(input_path, name)
    npz = np.load(fullname)

    predicted_class, actual_class = npz['predicted_class'], npz['actual_class']

    if predicted_class == 2:
        print('Label 2 is found')

    print(predicted_class, actual_class)

