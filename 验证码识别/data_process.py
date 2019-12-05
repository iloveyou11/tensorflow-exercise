from PIL import Image
from keras import backend as K
import random
import glob
import numpy as np
import matplotlib.pyplot as plt


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

# 读取训练集前100张图像
image = []
text = []
count = 0
for filename in glob.glob(TRAIN_DATA_DIR+'*.png'):
    image.append(np.array(Image.open(filename)))
    text.append(filename.lstrip(TRAIN_DATA_DIR).rstrip('.png')[1:])
    count += 1
    if count >= 100:
        break

# 数据可视化
# plt.figure()
# for i in range(20):
#     plt.subplot(5, 4, i+1)
#     plt.tight_layout()
#     plt.imshow(image[i])
#     plt.title("Label: {}".format(text[i]))
#     plt.xticks([])
#     plt.yticks([])
# plt.show()

image = np.array(image, dtype=np.float32)
# print(image.shape)  # (100, 60, 160, 3)

# 将RGB转化为灰度图


def rgb2grey(img):
    return np.dot(img[..., :3], [0.299, 0.587, 0.114])


image = rgb2grey(image)
# print(image.shape)  # (100, 60, 160)

# 数据可视化
# plt.figure()
# for i in range(20):
#     plt.subplot(5, 4, i+1)
#     plt.tight_layout()
#     plt.imshow(image[i], cmap='Greys')
#     plt.title("Label: {}".format(text[i]))
#     plt.xticks([])
#     plt.yticks([])
# plt.show()

# 数据规范化
image = image/255
# 适配keras图像数据格式


def fit_keras_channels(batch, rows=CAPTCHA_HEIGHT, cols=CAPTCHA_WIDTH):
    if K.image_data_format() == 'channels_first':
        batch = batch.reshape(batch.shape[0], 1, rows, cols)
        input_shape = (1, rows, cols)
    else:
        batch = batch.reshape(batch.shape[0], rows, cols, 1)
        input_shape = (rows, cols, 1)
    return batch, input_shape


image, input_shape = fit_keras_channels(image)
# print(image.shape)  # (100, 60, 160, 1)
# print(input_shape)  # (60, 160, 1)


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


text = list(text)
vec = [None]*len(text)
for i in range(len(vec)):
    vec[i] = text2vec(text[i])


# 将验证码向量解码为对应字符
def vec2text(vector):
    if not isinstance(vector, np.ndarray):
        vector = np.asarray(vector)
    vector = np.reshape(vector, [CAPTCHA_LEN, -1])
    text = ''
    for item in vector:
        text += CAPTCHA_CHARSET[np.argmax(item)]
    return text
