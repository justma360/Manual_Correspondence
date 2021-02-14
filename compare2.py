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

# # points from SIFT 
# pts1=np.load(file_base+'\keypoints.npy')
# pts2=np.load(file_base+'\keypoints2.npy')
# pts1 = pts1[:-1] # last point was out of boundry and i cbbs to fix boundary issues and add padding
# pts2 = pts2[:-1]


allPoints=[pts1, pts2]

window=151

img1KP=np.zeros(img1.shape)
img2KP=np.zeros(img2.shape)


circle_parameters = dict(thickness=3, # colour of the good connections
                   color=(255, 0, 0), # collour of all key points
                   radius=5
                   ) 


def calc_average(lst): 
    return sum(lst) / len(lst) 

print(allPoints[0][0])

descriptorMSEWindowDiff=[]
descriptorMSEPixelDiff=[]
descriptorWindowDiff=[]
descriptorPixelDiff=[]
matchTemplateScore=[]

for index in range(0,len(allPoints[0])):
  print('keypoint {0}'.format(index))

  windowIntensity=[]
  lowerIbound=[None]*2
  upperIbound=[None]*2
  lowerJbound=[None]*2
  upperJbound=[None]*2


  for keypointPairs,_ in enumerate(allPoints):
    lowerIbound[keypointPairs]=allPoints[keypointPairs][index][1]-math.floor(window/2) # I = rows = y = points(x,y) index 1
    upperIbound[keypointPairs]=allPoints[keypointPairs][index][1]+math.ceil(window/2)
    lowerJbound[keypointPairs]=allPoints[keypointPairs][index][0]-math.floor(window/2)
    upperJbound[keypointPairs]=allPoints[keypointPairs][index][0]+math.ceil(window/2)

    if lowerIbound[keypointPairs]<0:
      lowerIbound[keypointPairs]=0
    if upperIbound[keypointPairs]>=img1.shape[0]:
      upperIbound[keypointPairs]=img1.shape[0]
    if lowerJbound[keypointPairs]<0:
      lowerJbound[keypointPairs]=0
    if upperJbound[keypointPairs]>=img1.shape[1]:
      upperJbound[keypointPairs]=img1.shape[1]


  cv.circle(img1, (allPoints[0][index][0], allPoints[0][index][1]), **circle_parameters)
  img1KP[lowerIbound[0]:upperIbound[0] , lowerJbound[0]:upperJbound[0] , :]=img1[lowerIbound[0]:upperIbound[0] , lowerJbound[0]:upperJbound[0] , :]/255


  cv.circle(img2, (allPoints[1][index][0], allPoints[1][index][1]), **circle_parameters)
  img2KP[lowerIbound[1]:upperIbound[1] , lowerJbound[1]:upperJbound[1] ,:]=img2[lowerIbound[1]:upperIbound[1] , lowerJbound[1]:upperJbound[1] ,:]/255

  # (KP1 pixel - KP2 pixel)^2 . mean (pixel by pixel difference) 
  mse = (\
    np.square(\
    img1[lowerIbound[0]:upperIbound[0] , lowerJbound[0]:upperJbound[0] , 0]\
     - img2[lowerIbound[1]:upperIbound[1] , lowerJbound[1]:upperJbound[1] , 0])\
       ).mean(axis=None)

  mseWindow = np.square(\
    (img1[lowerIbound[0]:upperIbound[0] , lowerJbound[0]:upperJbound[0] , 0]).mean(axis=None)\
     - (img2[lowerIbound[1]:upperIbound[1] , lowerJbound[1]:upperJbound[1] , 0]).mean(axis=None))

  # return = mean of abs(arrayIMG1 - arrayIMG2)  difference in each pixel 
  pixelDiff = (abs(\
    ( img1[lowerIbound[0]:upperIbound[0] , lowerJbound[0]:upperJbound[0] , 0]\
      - img2[lowerIbound[1]:upperIbound[1] , lowerJbound[1]:upperJbound[1] , 0])\
        )).mean(axis=None)\

  # mean(KP1 window)  - mean(KP2 window)    (mean window by window difference)
  WindowDiff = abs(\
    (img1[lowerIbound[0]:upperIbound[0] , lowerJbound[0]:upperJbound[0] , 0]).mean(axis=None)\
      - (img2[lowerIbound[1]:upperIbound[1] , lowerJbound[1]:upperJbound[1] , 0]).mean(axis=None)\
        )

  
  score=cv.matchTemplate(img1[lowerIbound[0]:upperIbound[0] , lowerJbound[0]:upperJbound[0] , 0], img2[lowerIbound[0]:upperIbound[0] , lowerJbound[0]:upperJbound[0] , 0], cv.TM_SQDIFF_NORMED);


  descriptorMSEWindowDiff.append(mseWindow)
  descriptorMSEPixelDiff.append(mse)
  descriptorWindowDiff.append(WindowDiff)
  descriptorPixelDiff.append(pixelDiff)
  matchTemplateScore.append(score[0][0])

  

print(descriptorMSEWindowDiff)
print(descriptorMSEPixelDiff)
print(descriptorWindowDiff)
print(descriptorPixelDiff)
print(matchTemplateScore)

print('Average MSE WindowDif {0}'.format(calc_average(descriptorMSEWindowDiff)))
print('Average MSE PixelDif {0}'.format(calc_average(descriptorMSEPixelDiff)))
print('Average raw Window Diff {0}'.format(calc_average(descriptorWindowDiff)))
print('Average raw Pixel Diff {0}'.format(calc_average(descriptorPixelDiff)))
print('matchTemplate SQ Diff Normed {0}'.format(calc_average(matchTemplateScore)))



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