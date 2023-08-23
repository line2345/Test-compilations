import cv2
import numpy as np
import math

# 读取图像
image = cv2.imread('/home/ghlins/Test-compilations/enemy_blue_two.png')
orgin_image=image

#请选择你的英雄（误）
target = 'blue'
#简单阈值二值化处理对红色效果比较好，对蓝色效果很差，自适应阈值效果更差
if target == 'red':
    _ , image = cv2.threshold(cv2.split(image)[2],220,255,cv2.THRESH_BINARY)
#还是离谱，通道相减处理对蓝色效果很好，对红色效果很差
elif target == 'blue':
    image = cv2.subtract(cv2.split(image)[0], cv2.split(image)[2])

image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel=np.ones((5, 5), np.uint8))
image =cv2.GaussianBlur(image,(3,3),0) 
edges = cv2.Canny(image, 50, 150)

# 轮廓检测
area_threshold=100
filtered_contours=[]
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
for cont in contours:
    area=cv2.contourArea(cont)
    if area>area_threshold:
        x, y, w, h = cv2.boundingRect(cont)
        #在一个稍大的矩形范围内计算R和B的通道平均值
        av_red=cv2.mean(orgin_image[y-10:y+h+10, x-10:x+w+10])[2]
        av_blue=cv2.mean(orgin_image[y-10:y+h+10, x-10:x+w+10])[0]
        #筛选掉我方颜色和红蓝色特征不明显的区域（黑白色为主）和形状不符合的(很微妙的参数)
        if target == 'red':
            if av_red > av_blue and abs(av_red-av_blue)>20 and h>2.2*w:
                filtered_contours.append(cont)
                cv2.rectangle(orgin_image, (x, y), (x + w, y + h), (0, 0, 255), 2)
        if target == 'blue':
            if av_red < av_blue and abs(av_red-av_blue)>20 and h>2.2*w:
                filtered_contours.append(cont)
                cv2.rectangle(orgin_image, (x, y), (x + w, y + h), (0, 0, 255), 2)

#两两匹配灯条
#排一下序，确保后面轮廓是从左向右遍历的
filtered_contours = sorted(filtered_contours, key=lambda ctr: cv2.boundingRect(ctr)[0])
for i in range(len(filtered_contours) - 1):
    for j in range(i + 1, len(filtered_contours)):
        lx, ly, lw, lh = cv2.boundingRect(filtered_contours[i])
        rx, ry, rw, rh = cv2.boundingRect(filtered_contours[j])
        rec_x = lx + lw
        rec_y = ly - int(0.5*lh)  #灯管与装甲板高度大约为2:1
        rec_w = rx - lx - lw
        rec_h = lh + rh
        area = 0
        #我也不去匹配什么数字，平行了，大小近似了，既然那数字是白的那我就筛选两灯条能推测出来的装甲板上白色区域的面积了(是不是还不用管哨兵和塔的标志了？)
        for m in range(rec_x, rec_x + rec_w):
            for n in range(rec_y, rec_y + rec_h):
                b, g, r = orgin_image[n, m]
                if (b > 150 and g > 150 and r > 150):
                    area += 1
        #同时考虑到（小）装甲板长宽大约为1:1，所以在画面中无论车身如何旋转，高度至少大约大于等于宽度,筛去一部分(对于用的着两张图好像这一个条件就行了)
        if area > 0.25 * rec_h * rec_w and 1.1 * rec_h > rec_w:
            cv2.rectangle(orgin_image, (rec_x, rec_y), (rec_x + rec_w, rec_y + rec_h), (0, 255, 255), 2)
            cv2.circle(orgin_image, (int(rec_x + rec_w//2), int(rec_y + rec_h//2)), 10, (0, 255, 255), -1)

cv2.imshow('Armor Recognition', orgin_image)
cv2.waitKey(0)
