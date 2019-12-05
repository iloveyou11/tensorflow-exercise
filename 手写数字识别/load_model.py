from keras.models import load_model
import os
import numpy as np
from keras.utils import np_utils
from keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
# 规范化
X_test = x_test.reshape(10000, 784)
X_test = X_test.astype('float32')
X_test /= 255

n_classes = 10
Y_test = np_utils.to_categorical(y_test, n_classes)

save_dir = "./mnist/model/"
model_name = 'keras_mnist.h5'
model_path = os.path.join(save_dir, model_name)
mnist_model = load_model(model_path)

loss_and_metrics = mnist_model.evaluate(X_test, Y_test, verbose=2)
print("Test Loss: {}".format(loss_and_metrics[0]))
print("Test Accuracy: {}%".format(loss_and_metrics[1]*100))

predicted_classes = mnist_model.predict_classes(X_test)

correct = np.nonzero(predicted_classes == y_test)[0]
incorrect = np.nonzero(predicted_classes != y_test)[0]
print("Classified correctly count: {}".format(len(correct)))
print("Classified incorrectly count: {}".format(len(incorrect)))
