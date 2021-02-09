import cv2 as cv
import os 

file_base =os.getcwd()

print(file_base+"\images\IMG_6091.jpeg")

img1 = cv.imread(file_base+"\images\IMG_6031.jpeg", 0)
img2 = cv.imread(file_base+"\images\IMG_6027.jpeg", 0)
img1 = cv.cvtColor(img1,cv.COLOR_GRAY2BGR)
img2 = cv.cvtColor(img2,cv.COLOR_GRAY2BGR)

img=cv.hconcat([img1,img2])

#the [y, x] for each right-click event will be stored here
left_image = list()
right_image = list()

counter=0

#this function will be called whenever the mouse is right-clicked
def mouse_callback(event, x, y, flags, params):
  #right-click event value is 2
  if event == 2:
    global counter
    global left_image
    global right_image

    if (counter % 2) == 1:
      right_image.append((x, y)) #right side
      print("Right side image coord: {0} ".format(right_image))
      print('Action: Click LEFT side image point')
    else:
      left_image.append((x, y)) #left side
      print("Left side image coord: {0} ".format(left_image))
      print('Action: Click RIGHT side CORRESPONDING point')

    counter+=1


print('Image size: Vertical {0} x Horizontal {1}'.format(img1.shape[0],img1.shape[1]))
scale = 0.75 #change scale if the image is too big or too small
window_width = int(img.shape[1] * scale)
window_height = int(img.shape[0] * scale)
cv.namedWindow('image', cv.WINDOW_NORMAL)
cv.resizeWindow('image', window_width, window_height)

#set mouse callback function for window
cv.setMouseCallback('image', mouse_callback)
cv.imshow('image', img)
print("How to: Start by clicking A point on Left image then CORRESPONDING point on the RIGHT image")
cv.waitKey(0)
cv.destroyAllWindows()


print('\nLeftImages={0}'.format(left_image))
print('RightImages={0}'.format(right_image))

right_image2=[]
for index,value in enumerate(right_image):
  right_image2.append((value[1],value[0]-img1.shape[1]))


print('\npts1={0}'.format(left_image))
print('pts2={0}'.format(right_image2))

circle_parameters = dict(thickness=1, # colour of the good connections
                   color=(255, 0, 0), # collour of all key points
                   radius=3
                   ) 

for ind,coord in enumerate(left_image):
  # image = cv2.circle(image, center_coordinates, radius, color, thickness)
  cv.circle(img, (coord[0], coord[1]), **circle_parameters)
  cv.circle(img, (right_image[ind][0], right_image[ind][1]), **circle_parameters)

  cv.line(img, (left_image[ind]), (right_image[ind]), (0, 255, 0), thickness=2, lineType=8)

cv.namedWindow('img', cv.WINDOW_NORMAL)
cv.resizeWindow('img', window_width, window_height)
cv.imshow('img', img)
cv.waitKey(0)