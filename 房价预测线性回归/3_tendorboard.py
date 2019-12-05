import tensorflow as tf
import numpy as np
import pandas as pd


def normalize(df):
    return df.apply(lambda col: (col-col.mean())/col.std())


df = pd.read_csv('房价预测线性回归/data1.csv', names=['square', 'bedrooms', 'price'])
df = normalize(df)
ones = pd.DataFrame({'ones': np.ones(len(df))})
df = pd.concat([ones, df], axis=1)
# print(df.head())


# 数据处理
X_data = np.array(df[df.columns[0:3]])
y_data = np.array(df[df.columns[-1]]).reshape(len(df), 1)
print(X_data.shape, type(X_data))
print(y_data.shape, type(y_data))


# 创建显性回归模型
learning_rate = 0.01
epoch = 500
# 输入x y
with tf.name_scope('input'):
    X = tf.compat.v1.placeholder(tf.float32, X_data.shape)
    y = tf.compat.v1.placeholder(tf.float32, y_data.shape)
with tf.name_scope('hypothesis'):
    W = tf.compat.v1.get_variable(
        "weights", (X_data.shape[1], 1), initializer=tf.constant_initializer())
    y_pred = tf.matmul(X, W)
with tf.name_scope('loss'):
    loss_op = 1 / (2 * len(X_data)) * tf.matmul((y_pred - y),
                                                (y_pred - y), transpose_a=True)
with tf.name_scope('train'):
    train_op = tf.train.GradientDescentOptimizer(
        learning_rate=learning_rate).minimize(loss_op)


# 创建会话
with tf.compat.v1.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())
    writer = tf.compat.v1.summary.FileWriter('./summary', sess.graph)
    for e in range(1, epoch+1):
        sess.run(train_op, feed_dict={X: X_data, y: y_data})
        if e % 10 == 0:
            loss, w = sess.run([loss_op, W], feed_dict={X: X_data, y: y_data})
            print("Epoch %d \t Loss=%.4g \t Model: y = %.4gx1 + %.4gx2 + %.4g" %
                  (e, loss, w[1], w[2], w[0]))
writer.close()
