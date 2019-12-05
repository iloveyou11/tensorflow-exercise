from captcha.image import ImageCaptcha
import random
import numpy as np
import matplotlib.pyplot as plt
import PIL.Image as Image
import tensorflow as tf

# 定义常量
NUMBER = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
LOWERCASE = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u',
             'v', 'w', 'x', 'y', 'z']
UPPERCASE = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U',
             'V', 'W', 'X', 'Y', 'Z']

CAPTCHA_CHARSET = NUMBER   # 验证码字符集
CAPTCHA_LEN = 4            # 验证码长度
CAPTCHA_HEIGHT = 60        # 验证码高度
CAPTCHA_WIDTH = 160        # 验证码宽度

TRAIN_DATASET_SIZE = 5000     # 验证码数据集大小
TEST_DATASET_SIZE = 1000
TRAIN_DATA_DIR = './train-data/'  # 验证码数据集目录
TEST_DATA_DIR = './test-data/'


# 生成随机字符
def gen_random_text(charset=CAPTCHA_CHARSET, length=CAPTCHA_LEN):
    text = [random.choice(charset) for _ in range(length)]
    return ''.join(text)

# 创建并保存验证码数据集


def create_captcha_dataset(
        size=100,
        data_dir='./data/',
        height=60,
        width=160,
        image_format='.png'):
    if tf.io.gfile.exists(data_dir):
        tf.io.gfile.rmtree(data_dir)
    tf.io.gfile.makedirs(data_dir)

    captcha = ImageCaptcha(width=width, height=height)

    for _ in range(size):
        text = gen_random_text(CAPTCHA_CHARSET, CAPTCHA_LEN)
        captcha.write(text, data_dir+text+image_format)

    return None


# 训练集
create_captcha_dataset(TRAIN_DATASET_SIZE, TRAIN_DATA_DIR)
# 测试集
create_captcha_dataset(TEST_DATASET_SIZE, TEST_DATA_DIR)


def gen_captcha_dataset(
        size=100,
        height=60,
        width=160,
        image_format='.png'):
    captcha = ImageCaptcha(width=width, height=height)
    images, texts = [None]*size, [None]*size

    for i in range(size):
        texts[i] = gen_random_text(CAPTCHA_CHARSET, CAPTCHA_LEN)
        images[i] = np.array(Image.open(captcha.generate(texts[i])))

    return images, texts


# 生成100张验证码图像
images, texts = gen_captcha_dataset()


# 可视化验证码前20张图片
plt.figure()
for i in range(20):
    plt.subplot(5, 4, i+1)
    plt.tight_layout()
    plt.imshow(images[i])
    plt.title("Label: {}".format(texts[i]))
    plt.xticks([])
    plt.yticks([])
plt.show()
