
import tensorflow as tf
import os
from keras.layers.core import Dense, Activation
from keras.models import Sequential
from keras.utils import np_utils
import matplotlib.pyplot as plt
import numpy as np
from keras.datasets import mnist

# 获取数据集
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 规范化
X_train = x_train.reshape(60000, 784)
X_test = x_test.reshape(10000, 784)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

# 统计各标签数量
label, count = np.unique(y_train, return_counts=True)
# print(label, count)

# 可视化标签数量
fig = plt.figure()
plt.bar(label, count, width=0.7, align='center')
plt.title("Label Distribution")
plt.xlabel("Label")
plt.ylabel("Count")
plt.xticks(label)
plt.ylim(0, 7500)
for label, count in zip(label, count):
    plt.text(label, count, '%d' % count, ha='center', va='bottom', fontsize=10)
# plt.show()

# one-hot编码
n_classes = 10
# print('before one-hot:', y_train.shape)
Y_train = np_utils.to_categorical(y_train, n_classes)
# print('after one-hot:', Y_train.shape)
Y_test = np_utils.to_categorical(y_test, n_classes)

# 定义神经网络
model = Sequential()

model.add(Dense(512, input_shape=(784,)))
model.add(Activation('relu'))

model.add(Dense(512))
model.add(Activation('relu'))

model.add(Dense(10))
model.add(Activation('softmax'))

# 编译模型
model.compile(loss='categorical_crossentropy',
              metrics=['accuracy'], optimizer='adam')
# 开始训练
history = model.fit(
    X_train,
    Y_train,
    batch_size=128,
    epochs=5,
    verbose=2,
    validation_data=(X_test, Y_test)
)

# 可视化指标
# print(history.history)
fig = plt.figure()

plt.subplot(2, 1, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='lower right')

plt.subplot(2, 1, 2)
plt.plot(history.history['loss'])  # 损失
plt.plot(history.history['val_loss'])  # 测试集上的损失
plt.title('Model Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.tight_layout()
# plt.show()


# 保存模型
save_dir = "./mnist/model/"
if tf.io.gfile.exists(save_dir):
    tf.io.gfile.rmtree(save_dir)
tf.io.gfile.makedirs(save_dir)

model_name = 'keras_mnist.h5'
model_path = os.path.join(save_dir, model_name)
model.save(model_path)
print('Saved trained model at %s ' % model_path)
