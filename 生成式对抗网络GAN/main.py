# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os, imageio

# 加载手写数据集
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data')

# 定义参数
batch_size = 100 # 每次训练的样本数
z_dim = 100 # 输出大小
OUTPUT_DIR = 'samples' # 输出目录
if not os.path.exists(OUTPUT_DIR):
    os.mkdir(OUTPUT_DIR)

# 定义X/noise/is_training变量
X = tf.placeholder(dtype=tf.float32, shape=[None, 28, 28, 1], name='X')
noise = tf.placeholder(dtype=tf.float32, shape=[None, z_dim], name='noise')
is_training = tf.placeholder(dtype=tf.bool, name='is_training')

# 激活函数leakyRelu
def lrelu(x, leak=0.2):
    return tf.maximum(x, leak * x)

# 损失函数
def sigmoid_cross_entropy_with_logits(x, y):
    return tf.nn.sigmoid_cross_entropy_with_logits(logits=x, labels=y)


# 关键点---判别器/生成器的定义

# 1 判别器
def discriminator(image, reuse=None, is_training=is_training):
    momentum = 0.9 # 动量
    
    with tf.variable_scope('discriminator', reuse=reuse):

        # 卷积开始,filters越来越多,图片越来越小
        # h0: -1,28,28,1
        # h1: -1,24,24,64
        # h2: -1,12,12,128
        # h3: -1,6,6,256
        # h4: -1,3,3,512
        # h4作为判别器输出
        
        h0 = lrelu(tf.layers.conv2d(image, kernel_size=5, filters=64, strides=2, padding='same'))
        
        h1 = tf.layers.conv2d(h0, kernel_size=5, filters=128, strides=2, padding='same')
        # batch_norm转化为标准的高斯分布,指数加权滑动平均算法,decay是衰减系数
        h1 = lrelu(tf.contrib.layers.batch_norm(h1, is_training=is_training, decay=momentum))
        
        h2 = tf.layers.conv2d(h1, kernel_size=5, filters=256, strides=2, padding='same')
        h2 = lrelu(tf.contrib.layers.batch_norm(h2, is_training=is_training, decay=momentum))
        
        h3 = tf.layers.conv2d(h2, kernel_size=5, filters=512, strides=2, padding='same')
        h3 = lrelu(tf.contrib.layers.batch_norm(h3, is_training=is_training, decay=momentum))
        
        h4 = tf.contrib.layers.flatten(h3)
        h4 = tf.layers.dense(h4, units=1)
        # 返回经过sigmoid处理后的h4和未被激活的h4
        return tf.nn.sigmoid(h4), h4

# 2 生成器(输入的z是噪音,为二维tensor)
def generator(z, is_training=is_training):
    momentum = 0.9
    with tf.variable_scope('generator', reuse=None):
        d = 3

        # 逆卷积开始,filters越来越少
        # h0: -1,3,3,512
        # h1: -1,6,6,256
        # h2: -1,12,12,128
        # h3: -1,24,24,64
        # h4: -1,28,28,1
        # h4作为生成器的输出

        h0 = tf.layers.dense(z, units=d * d * 512)
        h0 = tf.reshape(h0, shape=[-1, d, d, 512])
        h0 = tf.nn.relu(tf.contrib.layers.batch_norm(h0, is_training=is_training, decay=momentum))
        
        h1 = tf.layers.conv2d_transpose(h0, kernel_size=5, filters=256, strides=2, padding='same')
        h1 = tf.nn.relu(tf.contrib.layers.batch_norm(h1, is_training=is_training, decay=momentum))
        
        h2 = tf.layers.conv2d_transpose(h1, kernel_size=5, filters=128, strides=2, padding='same')
        h2 = tf.nn.relu(tf.contrib.layers.batch_norm(h2, is_training=is_training, decay=momentum))
        
        h3 = tf.layers.conv2d_transpose(h2, kernel_size=5, filters=64, strides=2, padding='same')
        h3 = tf.nn.relu(tf.contrib.layers.batch_norm(h3, is_training=is_training, decay=momentum))
        
        h4 = tf.layers.conv2d_transpose(h3, kernel_size=5, filters=1, strides=1, padding='valid', activation=tf.nn.tanh, name='g')
        return h4

g = generator(noise) # 生成的假图片
d_real, d_real_logits = discriminator(X) # 真图片激活后h4和未激活h4的值
d_fake, d_fake_logits = discriminator(g, reuse=True) # 假图片激活后h4和未激活h4的值

vars_g = [var for var in tf.trainable_variables() if var.name.startswith('generator')] # 和generator相关的参数
vars_d = [var for var in tf.trainable_variables() if var.name.startswith('discriminator')] # 和discriminator相关的参数

loss_d_real = tf.reduce_mean(sigmoid_cross_entropy_with_logits(d_real_logits, tf.ones_like(d_real))) # 真图片导致的判别器损失
loss_d_fake = tf.reduce_mean(sigmoid_cross_entropy_with_logits(d_fake_logits, tf.zeros_like(d_fake))) # 假图片导致的判别器损失
loss_g = tf.reduce_mean(sigmoid_cross_entropy_with_logits(d_fake_logits, tf.ones_like(d_fake))) # 生成器损失
loss_d = loss_d_real + loss_d_fake # 判别器损失(真图片+假图片)


# 优化函数
# 先完成update_ops的相关操作(如BN的参数更新),再完成后续的优化操作
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    optimizer_d = tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.5).minimize(loss_d, var_list=vars_d)
    optimizer_g = tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.5).minimize(loss_g, var_list=vars_g)

# 辅助函数,用于将多张图片以网格状拼接在一起
def montage(images):
    if isinstance(images, list):
        images = np.array(images)
    img_h = images.shape[1]
    img_w = images.shape[2]
    n_plots = int(np.ceil(np.sqrt(images.shape[0])))
    m = np.ones((images.shape[1] * n_plots + n_plots + 1, images.shape[2] * n_plots + n_plots + 1)) * 0.5
    for i in range(n_plots):
        for j in range(n_plots):
            this_filter = i * n_plots + j
            if this_filter < images.shape[0]:
                this_img = images[this_filter]
                m[1 + i + i * img_h:1 + i + (i + 1) * img_h,
                  1 + j + j * img_w:1 + j + (j + 1) * img_w] = this_img
    return m


# 开始训练(需要交替训练,如每次迭代训练G两次)
sess = tf.Session()
sess.run(tf.global_variables_initializer())
z_samples = np.random.uniform(-1.0, 1.0, [batch_size, z_dim]).astype(np.float32)
samples = []
loss = {'d': [], 'g': []}

for i in range(60000):
    # 产生随机noise
    n = np.random.uniform(-1.0, 1.0, [batch_size, z_dim]).astype(np.float32)
    # 依次取数据
    batch = mnist.train.next_batch(batch_size=batch_size)[0]
    batch = np.reshape(batch, [-1, 28, 28, 1])
    # batch是0~1(relu),我们要将它映射到-1~1(tanh的取值范围)
    batch = (batch - 0.5) * 2 
    
    d_ls, g_ls = sess.run([loss_d, loss_g], feed_dict={X: batch, noise: n, is_training: True})
    loss['d'].append(d_ls)
    loss['g'].append(g_ls)
    
    #依次训练D-G-G(判别器训练1次,生成器训练2次)
    sess.run(optimizer_d, feed_dict={X: batch, noise: n, is_training: True})
    sess.run(optimizer_g, feed_dict={X: batch, noise: n, is_training: True})
    sess.run(optimizer_g, feed_dict={X: batch, noise: n, is_training: True})
    
    # 每迭代1000轮,打印样本
    if i % 1000 == 0:
        print(i, d_ls, g_ls)
        gen_imgs = sess.run(g, feed_dict={noise: z_samples, is_training: False})
        # -1~1转0~1
        gen_imgs = (gen_imgs + 1) / 2
        imgs = [img[:, :, 0] for img in gen_imgs]
        gen_imgs = montage(imgs)
        plt.axis('off')
        plt.imshow(gen_imgs, cmap='gray')
        plt.savefig(os.path.join(OUTPUT_DIR, 'sample_%d.jpg' % i))
        plt.show()
        samples.append(gen_imgs)

plt.plot(loss['d'], label='Discriminator')
plt.plot(loss['g'], label='Generator')
plt.legend(loc='upper right')
plt.savefig('Loss.png')
plt.show()
imageio.mimsave(os.path.join(OUTPUT_DIR, 'samples.gif'), samples, fps=5)

# 保存模型
saver = tf.train.Saver()
saver.save(sess, './mnist_dcgan', global_step=60000)