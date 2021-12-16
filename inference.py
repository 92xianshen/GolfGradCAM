import numpy as np
import tensorflow as tf

# Task
task = ('All', '645') # full time resolution
# task = ('All', '600')
# task = ('All', '550')
# task = ('All', '500')
# task = ('All', '450')

x_test = np.load('../dataset/normalized/X_test_norm.npz')['arr_0']
y_test = np.load('../dataset/normalized/y_test.npz')['arr_0']

# Sensor and length selections
sensor_set = {
    'All'   : [0, 1, 2, 3, 4, 5, 6, 7], # SG + Acc. + Gyro.
    'SG'    : [0, 1],                   # SG
    'Acc.'  : [2, 3, 4],                # Acc.
    'Gyro.' : [5, 6, 7],                # Gyro.
}

sample_set = {
    '645': [350, 995], # 645
    '600': [350, 950], # 600
    '550': [350, 900], # 550
    '500': [350, 850], # 500
    '450': [350, 800], # 450
}

sensor, sample = sensor_set[task[0]], sample_set[task[1]]

model = tf.keras.models.load_model('saved_model/convnet_{}_{}.h5'.format(task[0], task[1]))

x_testing, y_testing = x_test[..., sample[0]:sample[1], sensor], y_test

y_pred = []
for x in x_testing:
    y_pred += [model.predict(x[np.newaxis]).argmax()]
test_acc = np.mean(y_pred == y_test)
print('Test acc.: ', test_acc)