
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras import backend as K
import tensorflow as tf
import os
from keras.models import Sequential
from keras.utils import np_utils
import matplotlib.pyplot as plt
import numpy as np
from keras.datasets import mnist

# 获取数据集
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 规范化
img_rows, img_cols = 28, 28
if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

X_train = x_train.astype('float32')
X_test = x_test.astype('float32')
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

# 使用 Keras sequential model 定义 MNIST CNN 网络
model = Sequential()
# 第1层卷积，32个3x3的卷积核 ，激活函数使用 relu
model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu',
                 input_shape=input_shape))

# 第2层卷积，64个3x3的卷积核，激活函数使用 relu
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))

# 最大池化层，池化窗口 2x2
model.add(MaxPooling2D(pool_size=(2, 2)))

# Dropout 25% 的输入神经元
model.add(Dropout(0.25))

# 将 Pooled feature map 摊平后输入全连接网络
model.add(Flatten())

# 全联接层
model.add(Dense(128, activation='relu'))

# Dropout 50% 的输入神经元
model.add(Dropout(0.5))

# 使用 softmax 激活函数做多分类，输出各数字的概率
model.add(Dense(n_classes, activation='softmax'))

model.summary()

for layer in model.layers:
    print(layer.get_output_at(0).get_shape().as_list())

# 编译模型
model.compile(loss='categorical_crossentropy',
              metrics=['accuracy'], optimizer='adam')
# 训练模型
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
plt.show()


# 保存模型
save_dir = "./mnist/model/"
if tf.io.gfile.exists(save_dir):
    tf.io.gfile.rmtree(save_dir)
tf.io.gfile.makedirs(save_dir)

model_name = 'keras_mnist.h5'
model_path = os.path.join(save_dir, model_name)
model.save(model_path)
print('Saved trained model at %s ' % model_path)
