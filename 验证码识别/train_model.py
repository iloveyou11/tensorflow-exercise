from PIL import Image
from keras import backend as K
from keras.utils.vis_utils import plot_model
from keras.models import *
from keras.layers import *

import glob
import pickle

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


NUMBER = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
LOWERCASE = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u',
             'v', 'w', 'x', 'y', 'z']
UPPERCASE = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U',
             'V', 'W', 'X', 'Y', 'Z']

CAPTCHA_CHARSET = NUMBER   # 验证码字符集
CAPTCHA_LEN = 4            # 验证码长度
CAPTCHA_HEIGHT = 60        # 验证码高度
CAPTCHA_WIDTH = 160        # 验证码宽度

TRAIN_DATA_DIR = './train-data/'  # 验证码数据集目录
TEST_DATA_DIR = './test-data/'

BATCH_SIZE = 100
EPOCHS = 10
OPT = 'adam'
LOSS = 'binary_crossentropy'

MODEL_DIR = './model/train_demo/'
MODEL_FORMAT = '.h5'
HISTORY_DIR = './history/train_demo/'
HISTORY_FORMAT = '.history'

filename_str = "{}captcha_{}_{}_bs_{}_epochs_{}{}"

# 模型网络结构文件
MODEL_VIS_FILE = 'captcha_classfication' + '.png'
# 模型文件
MODEL_FILE = filename_str.format(
    MODEL_DIR, OPT, LOSS, str(BATCH_SIZE), str(EPOCHS), MODEL_FORMAT)
# 训练记录文件
HISTORY_FILE = filename_str.format(
    HISTORY_DIR, OPT, LOSS, str(BATCH_SIZE), str(EPOCHS), HISTORY_FORMAT)


def rgb2gray(img):
    return np.dot(img[..., :3], [0.299, 0.587, 0.114])


# 对验证码中每个字符进行 one-hot 编码


def text2vec(text, length=CAPTCHA_LEN, charset=CAPTCHA_CHARSET):
    text_len = len(text)
    # 验证码长度校验
    if text_len != length:
        raise ValueError(
            'Error: length of captcha should be {}, but got {}'.format(length, text_len))

    # 生成一个形如（CAPTCHA_LEN*CAPTHA_CHARSET,) 的一维向量
    # 例如，4个纯数字的验证码生成形如(4*10,)的一维向量
    vec = np.zeros(length * len(charset))
    for i in range(length):
        # One-hot 编码验证码中的每个数字
        # 每个字符的热码 = 索引 + 偏移量
        vec[charset.index(text[i]) + i*len(charset)] = 1
    return vec

# 将验证码向量解码为对应字符


def vec2text(vector):
    if not isinstance(vector, np.ndarray):
        vector = np.asarray(vector)
    vector = np.reshape(vector, [CAPTCHA_LEN, -1])
    text = ''
    for item in vector:
        text += CAPTCHA_CHARSET[np.argmax(item)]
    return text


def fit_keras_channels(batch, rows=CAPTCHA_HEIGHT, cols=CAPTCHA_WIDTH):
    if K.image_data_format() == 'channels_first':
        batch = batch.reshape(batch.shape[0], 1, rows, cols)
        input_shape = (1, rows, cols)
    else:
        batch = batch.reshape(batch.shape[0], rows, cols, 1)
        input_shape = (rows, cols, 1)
    return batch, input_shape

# 读取训练集


X_train = []
Y_train = []
for filename in glob.glob(TRAIN_DATA_DIR + '*.png'):
    X_train.append(np.array(Image.open(filename)))
    Y_train.append(filename.lstrip(TRAIN_DATA_DIR).rstrip('.png')[1:])
X_train = np.array(X_train, dtype=np.float32)
X_train = rgb2gray(X_train)
X_train = X_train / 255
X_train, input_shape = fit_keras_channels(X_train)
# (3948, 60, 160, 1) <class 'numpy.ndarray'>
print(X_train.shape, type(X_train))
print(input_shape)  # (60, 160, 1)

# 处理训练集标签
Y_train = list(Y_train)
for i in range(len(Y_train)):
    Y_train[i] = text2vec(Y_train[i])
Y_train = np.asarray(Y_train)
print(Y_train.shape, type(Y_train))


# 读取测试集，处理对应图像和标签
X_test = []
Y_test = []
for filename in glob.glob(TEST_DATA_DIR + '*.png'):
    X_test.append(np.array(Image.open(filename)))
    Y_test.append(filename.lstrip(TEST_DATA_DIR).rstrip('.png')[1:])
# list -> rgb -> gray -> normalization -> fit keras
X_test = np.array(X_test, dtype=np.float32)
X_test = rgb2gray(X_test)
X_test = X_test / 255
X_test, _ = fit_keras_channels(X_test)
Y_test = list(Y_test)
for i in range(len(Y_test)):
    Y_test[i] = text2vec(Y_test[i])
Y_test = np.asarray(Y_test)
print(X_test.shape, type(X_test))
print(Y_test.shape, type(Y_test))


# 创建验证码识别模型
inputs = Input(shape=input_shape, name="inputs")
# 第1层卷积
conv1 = Conv2D(32, (3, 3), name="conv1")(inputs)
relu1 = Activation('relu', name="relu1")(conv1)

# 第2层卷积
conv2 = Conv2D(32, (3, 3), name="conv2")(relu1)
relu2 = Activation('relu', name="relu2")(conv2)
pool2 = MaxPooling2D(pool_size=(2, 2), padding='same', name="pool2")(relu2)

# 第3层卷积
conv3 = Conv2D(64, (3, 3), name="conv3")(pool2)
relu3 = Activation('relu', name="relu3")(conv3)
pool3 = MaxPooling2D(pool_size=(2, 2), padding='same', name="pool3")(relu3)

# 将 Pooled feature map 摊平后输入全连接网络
x = Flatten()(pool3)

# Dropout
x = Dropout(0.25)(x)

# 4个全连接层分别做10分类，分别对应4个字符。
x = [Dense(10, activation='softmax', name='fc%d' % (i+1))(x) for i in range(4)]

# 4个字符向量拼接在一起，与标签向量形式一致，作为模型输出。
outs = Concatenate()(x)

# 定义模型的输入与输出
model = Model(inputs=inputs, outputs=outs)
model.compile(optimizer=OPT, loss=LOSS, metrics=['accuracy'])

model.summary()
plot_model(model, to_file=MODEL_VIS_FILE, show_shapes=True)
history = model.fit(X_train,
                    Y_train,
                    batch_size=BATCH_SIZE,
                    epochs=EPOCHS,
                    verbose=2,
                    validation_data=(X_test, Y_test))

print(vec2text(Y_test[9]))
yy = model.predict(X_test[9].reshape(1, 60, 160, 1))
print(vec2text(yy))

if not tf.io.gfile.exists(MODEL_DIR):
    tf.io.gfile.makedirs(MODEL_DIR)
model.save(MODEL_DIR)
print('Saved trained model at %s ' % MODEL_FILE)
