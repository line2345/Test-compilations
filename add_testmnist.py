import numpy as np
import cv2
import struct
import tensorflow as tf
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Flatten

# 新增加的图像文件名和对应标签
image_files = ['test_3.png', 'test_4.png', 'test_6.png', 'test_7.png']
labels = [3, 4, 6, 7]
# 用于存放图像和标签的列表
images = []
all_labels = []
# 遍历图像文件和标签
for i, image_file in enumerate(image_files):
    # 读取图像并进行灰度化
    image = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (28, 28))
    image = image / 255.0
    images.append(image)
    all_labels.append(labels[i])
# 将图像列表和标签列表转换为NumPy数组
images = np.array(images)
labels = np.array(all_labels)
# 创建一个空的数据集数组，形状为(图像数量, 图像宽度, 图像高度)
dataset = np.empty((images.shape[0], images.shape[1], images.shape[2]), dtype=np.float32)
# 将图像数组复制到数据集数组中
dataset[:, :, :] = images
# 将标签数组复制到数据集的标签中
dataset_labels = labels

class MnistData: #梅开二度
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

mnist_data = MnistData("train-images.idx3-ubyte", "train-labels.idx1-ubyte", "t10k-images.idx3-ubyte", "t10k-labels.idx1-ubyte")
# 将新增加的图像和标签添加到原始训练集中，顺便将原标签进行one-hot
new_train_images = np.concatenate([mnist_data.train_images, dataset], axis=0)
new_train_labels = np.concatenate([mnist_data.train_labels, np.eye(10)[dataset_labels]], axis=0)


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
history = model.fit(new_train_images,new_train_labels,epochs=10,validation_data=(mnist_data.test_images,mnist_data.test_labels))
model.evaluate(mnist_data.test_images,mnist_data.test_labels)
model.save('new_mnist_model.h5')