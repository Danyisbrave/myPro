import numpy as np
import glob
import os

def load_data(root, vfold_ratio=0.2, max_items_per_class=20):
    all_files = glob.glob(os.path.join(root, '*.npy'))

    # initialize variables
    x = np.empty([0, 784])
    y = np.empty([0])
    class_names = []

    # load a subset of the data to memory
    for idx, file in enumerate(all_files):
        data = np.load(file)
        data = data[0: max_items_per_class, :]
        labels = np.full(data.shape[0], idx)

        x = np.concatenate((x, data), axis=0)
        y = np.append(y, labels)

        class_name, ext = os.path.splitext(os.path.basename(file))
        class_names.append(class_name)

    data = None
    labels = None

    # separate into training and testing
    permutation = np.random.permutation(y.shape[0])
    x = x[permutation, :]
    y = y[permutation]

    vfold_size = int(x.shape[0] / 100 * (vfold_ratio * 100))

    x_test = x[0:vfold_size, :]
    y_test = y[0:vfold_size]

    x_train = x[vfold_size:x.shape[0], :]
    y_train = y[vfold_size:y.shape[0]]
    return x_train, y_train, x_test, y_test, class_names

load_data('D:\\迅雷下载\\npy')