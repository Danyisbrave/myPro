from kerasTest import dataRead
from tensorflow import keras


def dataprocess(path, max_items_per_class):
    x_train, y_train, x_test, y_test, class_names = dataRead.load_data(path, max_items_per_class=max_items_per_class)

    num_classes = len(class_names)
    image_size = 28

    # Reshape and normalize
    x_train = x_train.reshape(x_train.shape[0], image_size, image_size, 1).astype('float32')
    x_test = x_test.reshape(x_test.shape[0], image_size, image_size, 1).astype('float32')

    x_train /= 255.0
    x_test /= 255.0

    # Convert class vectors to class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    return x_train, y_train, x_test, y_test
