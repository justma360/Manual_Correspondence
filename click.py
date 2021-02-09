import cv2 as cv

file_base = "C:/Users/Justin/Documents/Work/Imperial_College_London/Projects/CVPR_CW1/images/"

img1 = cv.imread(file_base+"IMG_6091.jpeg", 0)
img2 = cv.imread(file_base+"IMG_6100.jpeg", 0)
img1 = cv.cvtColor(img1,cv.COLOR_GRAY2BGR)
img2 = cv.cvtColor(img2,cv.COLOR_GRAY2BGR)

img=cv.hconcat([img1,img2])

#the [y, x] for each right-click event will be stored here
right_clicks = []

#this function will be called whenever the mouse is right-clicked
def mouse_callback(event, x, y, flags, params):
    #right-click event value is 2
    if event == 2:
        global right_clicks
        if not right_clicks:
          
        #store the coordinates of the right-click event
        right_clicks.append([y, x])

        #this just verifies that the mouse data is being collected
        #you probably want to remove this later
        print(right_clicks)

print('Image size: Vertical {0} x Horizontal {1}'.format(img1.shape[0],img1.shape[1]))
scale = 0.5
window_width = int(img.shape[1] * scale)
window_height = int(img.shape[0] * scale)
cv.namedWindow('image', cv.WINDOW_NORMAL)
cv.resizeWindow('image', window_width, window_height)

#set mouse callback function for window
cv.setMouseCallback('image', mouse_callback)
cv.imshow('image', img)
print("Start by clicking Left image")
cv.waitKey(0)
cv.destroyAllWindows()

# Left Image
def even_indices(lst):
  new_lst = []
  for index in range(0, len(lst), 2):
    new_lst.append(lst[index])
  return new_lst

# Right Image
def odd_indices(lst):
  new_lst = []
  for index in range(1, len(lst), 2):
    new_lst.append(lst[index])
  return new_lst

pts1=even_indices(right_clicks)
pts2=odd_indices(right_clicks)

print('LeftImages Merged',pts1)
print('RightImages Merged',pts2)

pts1=pts1
# remove the horizontal shift
for index,value in enumerate(pts2):
  pts2[index][1]=pts2[index][1]-img1.shape[1] 

print('LeftImages individual',pts1)
print('RightImages individual',pts2)

