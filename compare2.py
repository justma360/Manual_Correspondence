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


# coord of the key points from manual selection
pts1=[(697, 1026), (698, 778), (1258, 884), (1425, 872), (1238, 778), (953, 1001), (1029, 1042), (1342, 1097), (1174, 677)]
pts2=[(580, 842), (728, 613), (1117, 816), (1306, 844), (1173, 708), (805, 868), (868, 922), (1136, 1053), (1193, 586)]

#points are stored as (columns,rows) (x,y) 

allPoints=[pts1, pts2]

window=151

img1KP=np.zeros(img1.shape)
img2KP=np.zeros(img2.shape)
print(img1KP.shape)



circle_parameters = dict(thickness=3, # colour of the good connections
                   color=(255, 0, 0), # collour of all key points
                   radius=5
                   ) 


def calc_average(lst): 
    return sum(lst) / len(lst) 


descriptor=[]
for index,kPSets in enumerate(allPoints): 
  descriptor.append([])
  for keyPoints in kPSets:
    print('keypoint location {0}'.format(keyPoints))
    # for the x coord within 5 window
    windowIntensity=[]
    lowerIbound=keyPoints[1]-math.floor(window/2) # I = rows = y = points(x,y) index 1
    upperIbound=keyPoints[1]+math.ceil(window/2)
    lowerJbound=keyPoints[0]-math.floor(window/2)
    upperJbound=keyPoints[0]+math.ceil(window/2)
    if lowerIbound<0:
      lowerIbound=0
    if upperIbound>=img1.shape[0]:
      upperIbound=img1.shape[0]
    if lowerJbound<0:
      lowerJbound=0
    if upperJbound>=img1.shape[1]:
      upperJbound=img1.shape[1]


    if index == 0:
      cv.circle(img1, (keyPoints[0], keyPoints[1]), **circle_parameters)
      img1KP[lowerIbound:upperIbound,lowerJbound:upperJbound,:]=img1[lowerIbound:upperIbound,lowerJbound:upperJbound,:]/255
      windowIntensity=np.average(img1[lowerIbound:upperIbound,lowerJbound:upperJbound,0])

    elif index== 1:
      cv.circle(img2, (keyPoints[0], keyPoints[1]), **circle_parameters)
      img2KP[lowerIbound:upperIbound,lowerJbound:upperJbound,:]=img2[lowerIbound:upperIbound,lowerJbound:upperJbound,:]/255
      windowIntensity=np.average(img2[lowerIbound:upperIbound,lowerJbound:upperJbound,0])

    #find average intensity of each window
    descriptor[index].append((windowIntensity))

print(descriptor)
diff=[]
for index in range(0,len(descriptor[0])): 
  diff.append(abs(descriptor[0][index]-descriptor[1][index]))

print(diff)
print('Average Difference {0}'.format(calc_average(diff)))

# Plotting the images 
fig, axes = plt.subplots(ncols=2,figsize=(20,10))
fig.subplots_adjust(hspace=0.1)
axes[0].set_title('Image from the left side keypoints')
axes[0].imshow(img1KP,cmap='gray',vmin=0, vmax=256)
axes[0].set_axis_off()

axes[1].set_title('Image from the right side keypoints')
axes[1].imshow(img2KP,cmap='gray_r',vmin=0, vmax=256)
axes[1].set_axis_off()

plt.show()

