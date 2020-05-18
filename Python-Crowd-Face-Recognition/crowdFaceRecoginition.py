import cv2
import numpy as np
from PIL import Image

#create image
image = np.zeros((400,400,3), np.uint8)
img_array = []

face_cascade = cv2.CascadeClassifier('C:/abhi/cv2/haarcascade_frontalface_default.xml')
#cascPath = 'haarcascade_frontalface_default.xml'
#faceCascade = cv2.CascadeClassifier(cascPath)
# Read the input image
#img = cv2.imread('test.png')

cap = cv2.VideoCapture('C:/abhi/cv2/test.mp4')

# Get the Default resolutions
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

# Define the codec and filename.
out = cv2.VideoWriter('C:/abhi/cv2/output.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))
out1 = cv2.VideoWriter('C:/abhi/cv2/output1.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))
out2 = cv2.VideoWriter('C:/abhi/cv2/output3.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (50,50))

while cap.isOpened():
    ret, img = cap.read()
    #print (ret)
    #cv2.imshow('img', img)
    if ret==True:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray,  1.1, 4)
        flag=False

        for (x, y , w ,h) in faces:
            cv2.rectangle(img, (x,y), (x+w, y+h), (255, 0 , 0), 3)
            img2 = img[y:y+h, x:x+w]
            img_array.append(img2)
            #cv2.imshow('img2', img2)
            img3 = cv2.resize(img2, (50,50), interpolation = cv2.INTER_NEAREST)
            #crop image
            #crop = image[100:300,100:300]
            #tf.image.crop_to_bounding_box(            img[y:y+h, x:x+w], offset_height, offset_width, target_height, target_width            )
            
            out2.write(img3)         
            flag=True
            
            # Display the output
            #cv2.imshow('img2', img2)

        if flag==True:
            
            out1.write(img) 
            
        out.write(img)
       
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break
cap.release()
out.release()
out1.release()
out2.release()

i=0
for x in img_array:
    path = 'C:/abhi/cv2/output/'
    i +=1
    cv2.imwrite(str(path) + str(i) + '.jpg', x)


