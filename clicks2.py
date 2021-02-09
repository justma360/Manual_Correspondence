import cv2 as cv

file_base = "C:/Users/Justin/Documents/Work/Imperial_College_London/Projects/CVPR_CW1/images/"

img1 = cv.imread(file_base+"IMG_6091.jpeg", 0)
img2 = cv.imread(file_base+"IMG_6100.jpeg", 0)
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
      right_image.append([y, x-img1.shape[1]])
      print("Right side image coord: {0} ".format(right_image))
      print('Action: Click LEFT side image point')
    else:
      left_image.append([y, x])
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


print('\nLeftImages individual: {0}'.format(left_image))
print('RightImages individual: {0}'.format(right_image))
