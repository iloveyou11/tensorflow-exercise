# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

batch_size = 100
z_dim = 100

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

sess = tf.Session()
sess.run(tf.global_variables_initializer())

saver = tf.train.import_meta_graph('./mnist_dcgan-60000.meta')
saver.restore(sess, tf.train.latest_checkpoint('./'))

graph = tf.get_default_graph()
g = graph.get_tensor_by_name('generator/g/Tanh:0')
noise = graph.get_tensor_by_name('noise:0')
is_training = graph.get_tensor_by_name('is_training:0')

n = np.random.uniform(-1.0, 1.0, [batch_size, z_dim]).astype(np.float32)
gen_imgs = sess.run(g, feed_dict={noise: n, is_training: False})
gen_imgs = (gen_imgs + 1) / 2
imgs = [img[:, :, 0] for img in gen_imgs]
gen_imgs = montage(imgs)
plt.axis('off')
plt.imshow(gen_imgs, cmap='gray')
plt.show()