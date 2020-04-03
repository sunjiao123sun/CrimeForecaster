import numpy as np
import os

#Chicago
feature_file_name = 'feature_generated.npy'
label_file_name = 'label_generated.npy'
x = np.load(feature_file_name, allow_pickle=True)
y = np.load(label_file_name, allow_pickle=True)
y = np.expand_dims(y, axis = 1)

num_samples = x.shape[0]
num_test = round(num_samples * 0.2)
num_train = round(num_samples * 0.7)
num_val = num_samples - num_test - num_train

# output_dir = '/home/users/jiaosun/DCRNN/data/CRIME-LA/'
output_dir = '/home/users/jiaosun/DCRNN/data/CRIME-CHICAGO/'
# train
x_train, y_train = x[:num_train], y[:num_train]
# val
x_val, y_val = (
    x[num_train: num_train + num_val],
    y[num_train: num_train + num_val],
)
# test
x_test, y_test = x[-num_test:], y[-num_test:]

x_offsets = np.array([-11, -10,  -9,  -8,  -7,  -6,  -5,  -4,  -3,  -2,  -1,   0])
y_offsets = np.array([ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12])

for cat in ["train", "val", "test"]:
    _x, _y = locals()["x_" + cat], locals()["y_" + cat]
    print(cat, "x: ", _x.shape, "y:", _y.shape)
    np.savez_compressed(
        os.path.join(output_dir, "%s.npz" % cat),
        x=_x,
        y=_y,
        x_offsets=x_offsets.reshape(list(x_offsets.shape) + [1]),
        y_offsets=y_offsets.reshape(list(y_offsets.shape) + [1]),
    )
