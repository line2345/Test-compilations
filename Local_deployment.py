import tensorflow as tf
import cv2
import os
import numpy as np

loaded_model = tf.keras.models.load_model('new_mnist_model.h5')
for test in ['./test_3.png', './test_4.png', './test_6.png', './test_7.png']:
    image = cv2.imread(test, 0)
    image = cv2.resize(image, (28, 28))

    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    prediction = loaded_model.predict(image)
    predicted_label = np.argmax(prediction)

    print('预测结果:', predicted_label)