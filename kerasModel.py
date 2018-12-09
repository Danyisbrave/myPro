from keras.models import Sequential
from tensorflow import python_io
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF

from keras.layers import Conv2D, MaxPooling2D, Flatten, Convolution2D
from keras.layers import Dropout, Dense

from kerasTest import dataProcess
KTF.set_session(tf.Session(config=tf.ConfigProto(device_count={'gpu':0})))

python_io.tf_record_iterator('rnn_tutorrial_data/eval.tfrecord-00000-of-00010')


# 训练数据
x_train, y_train, x_test, y_test = dataProcess.dataprocess(path='D:\\迅雷下载\\npy',max_items_per_class=500)

# 分类器，分类数目
num_classes = y_train.shape[1]
# 定义图片大小
input_shape = x_train.shape[1:]

# 每次梯度更新的样本数
batch_size = 128


# epochs: 整数。训练模型迭代轮次。一个轮次是在整个 x 和 y 上的一轮迭代。
epochs = 12



model = Sequential()

model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(units=num_classes, activation='softmax'))

# categorical_crossentropy 计算交叉熵
# Adadelta优化器:Adadelta是Adagrad的一个具有更强鲁棒性的的扩展版本，它不是累积所有过去的梯度，而是根据渐变更新的移动窗口调整学习速率。
# 这样，即使进行了许多更新，Adadelta仍在继续学习。 与Adagrad相比，在Adadelta的原始版本中，您无需设置初始学习率。
# 在此版本中，与大多数其他Keras优化器一样，可以设置初始学习速率和衰减因子。
# 评估标准 metrics
# model.compile(loss='categorical_crossentropy',
#               optimizer=keras.optimizers.Adadelta(),
#               metrics=['top_k_categorical_accuracy'])
model.compile(loss='categorical_crossentropy',
              optimizer=tf.train.Adadelta(),
              metrics=['top_k_categorical_accuracy'])

model.fit(x=x_train, y=y_train, batch_size=batch_size, epochs=epochs, verbose=1)

model.save('my_model.h5')
score = model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
