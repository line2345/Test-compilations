import numpy as np
import struct
import tensorflow as tf
from keras.datasets import mnist #整不下来emmmm，无可奈何只能class MnistData
from keras.models import Sequential
from keras.layers import Dense, Flatten 
import matplotlib.pyplot as plt

class MnistData:
    def __init__(self, train_image_path, train_label_path, test_image_path, test_label_path):
        # 训练集和验证集的文件路径
        self.train_image_path = train_image_path
        self.train_label_path = train_label_path
        self.test_image_path = test_image_path
        self.test_label_path = test_label_path

        # 获取训练集和验证集数据
        # get_data()方法，若参数为0则获取训练集数据，若参数为1则获取验证集
        self.train_images, self.train_labels = self.get_data(0)
        self.test_images, self.test_labels = self.get_data(1)

    def get_data(self, data_type):
        if data_type == 0:  # 获取训练集数据
            image_path = self.train_image_path
            label_path = self.train_label_path
        else:  # 获取验证集数据
            image_path = self.test_image_path
            label_path = self.test_label_path
        
        with open(image_path, 'rb') as file1:
            image_file = file1.read()
        with open(label_path, 'rb') as file2:
            label_file = file2.read()

        image_index = 0
        label_index = 0
        labels = []
        images = []

        # 读取训练集图片数据文件的文件信息
        magic, num_of_datasets, rows, columns = struct.unpack_from('>IIII', image_file, image_index)
        image_index += struct.calcsize('>IIII')

        for i in range(num_of_datasets):
            # 读取784个unsigned byte，即一张图片的所有像素值
            temp = struct.unpack_from('>784B', image_file, image_index)
            # 将读取的像素数据转换成28×28的矩阵
            temp = np.reshape(temp, (28, 28))
            # 归一化处理
            temp = temp / 255
            images.append(temp)
            image_index += struct.calcsize('>784B')  # 每次增加784B

        # 跳过描述信息
        label_index += struct.calcsize('>II')
        labels = struct.unpack_from('>' + str(num_of_datasets) + 'B', label_file, label_index)
        # 独热向量编码
        labels = np.eye(10)[np.array(labels)]

        return np.array(images), labels
        #读文件部分完全看不懂

def plot_history(history):
    # 绘制训练集和验证集的损失曲线
    plt.figure()
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.legend(['Training', 'Validation'])

    # 检查准确度指标是否存在
    if 'accuracy' in history.history:
        # 绘制训练集和验证集的准确度曲线
        plt.figure()
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.legend(['Training', 'Validation'])

    plt.show()
    #可视化，剽的(划去)，好的现在不是剽的了，原来超过头了根本用不成

mnist_data = MnistData("train-images.idx3-ubyte", "train-labels.idx1-ubyte", "t10k-images.idx3-ubyte", "t10k-labels.idx1-ubyte")
model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(512, activation=tf.nn.relu),
    tf.keras.layers.Dropout(0.2),
    Dense(10, activation=tf.nn.softmax)
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.summary()
history = model.fit(mnist_data.train_images,mnist_data.train_labels,epochs=10,validation_data=(mnist_data.test_images,mnist_data.test_labels))
model.evaluate(mnist_data.test_images,mnist_data.test_labels)
plot_history(history)
model.save('mnist_model.h5')
