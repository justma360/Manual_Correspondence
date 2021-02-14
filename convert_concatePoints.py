import cv2 as cv
import os 
import math
import matplotlib.pyplot as plt
import numpy as np

# reopen images 
file_base =os.getcwd()
print(file_base+"\images\IMG_6091.jpeg")
img1 = cv.imread(file_base+"\images\IMG_6031.jpeg", 0)
img2 = cv.imread(file_base+"\images\IMG_6027.jpeg", 0)
img1 = cv.cvtColor(img1,cv.COLOR_GRAY2BGR)
img2 = cv.cvtColor(img2,cv.COLOR_GRAY2BGR)

print(img1.shape[1])

# pts1=[(697, 1026), (698, 778), (1258, 884), (1425, 872), (1238, 778), (953, 1001), (1029, 1042), (1342, 1097), (1174, 677)]
# pts2=[(580, 842), (728, 613), (1117, 816), (1306, 844), (1173, 708), (805, 868), (868, 922), (1136, 1053), (1193, 586)]


# points from SIFT 
pts1=np.load(file_base+'\keypoints.npy')
pts2=np.load(file_base+'\keypoints2.npy')
pts1 = pts1[:-1] # last point was out of boundry and i cbbs to fix boundary issues and add padding
pts2 = pts2[:-1]

pts1concate=[]
pts2concate=[]

for index,value in enumerate(pts1):
  pts1concate.append(tuple(value))

for index,value in enumerate(pts2):
  concateX=pts2[index][0] + img1.shape[1]
  pts2concate.append( (concateX,pts2[index][1]) )


print('\nLeftImages={0}'.format(pts1concate))
print('RightImages={0}'.format(pts2concate))