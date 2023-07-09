import cv2
import numpy as np

target_color="blue"
color_dist = {'red': {'Lower': np.array([0, 60, 60]), 'Upper': np.array([6, 255, 255])},
              'blue': {'Lower': np.array([100, 80, 46]), 'Upper': np.array([124, 255, 255])},
              'green': {'Lower': np.array([35, 43, 35]), 'Upper': np.array([90, 255, 255])},
              }
image = cv2.imread('indoor.jpg')
image=cv2.GaussianBlur(image,(3,3),0)
image=cv2.dilate(image,None,iterations=4)
image=cv2.erode(image,None,iterations=1)
image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
erode_image=cv2.erode(image_hsv,None,iterations=2)
dilate_image=cv2.dilate(erode_image,None,iterations=1)
inRange_image=cv2.inRange(dilate_image,color_dist[target_color]['Lower'],color_dist[target_color]['Upper'])
contours,_=cv2.findContours(inRange_image,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
area_threshold=400
for cont in contours:
    area=cv2.contourArea(cont)
    if area>area_threshold:
        cv2.drawContours(image, [cont], -1, (255, 255, 255))
        M=cv2.moments(cont)
        if M["m00"] != 0:
            cX=int(M["m10"] / M["m00"])
            cY=int(M["m01"] / M["m00"])
            cv2.circle(image, (cX, cY), 5, (0, 0, 255), -1)
            print("(%d %d)"%(cX,cY))
cv2.imshow("image",image)
cv2.waitKey()
