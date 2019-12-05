# 获取mnist数据集
import matplotlib.pyplot as plt
from keras.datasets import mnist

(train_x, train_y), (test_x, test_y) = mnist.load_data()
print(train_x.shape, train_y.shape)
print(test_x.shape, test_y.shape)

# 可视化数据集
fig = plt.figure()
for i in range(15):
    plt.subplot(3, 5, i+1)
    plt.tight_layout()
    plt.imshow(train_x[i], cmap='Greys')
    plt.title('label:{}'.format(train_y[i]))
    plt.xticks([])
    plt.yticks([])
plt.show()
