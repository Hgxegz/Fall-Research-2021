import cv2 as cv

# Loading the Cascade as well as reading in the input image 
# Hello
face_cascade = cv.CascadeClassifier('haarcascade_frontalcatface.xml')
img = cv.imread('face-images/Mona_Lisa.jpg')

# Scaling the image to reduce the size, will write a function in future
scalePercent = 40
width = int(img.shape[1] * scalePercent / 100)
height = int(img.shape[0] * scalePercent / 100)
dim = (width, height)
resizedImg = cv.resize(img, dim, interpolation = cv.INTER_AREA)
# Convert the resized image into grayscale 
grayImg = cv.cvtColor(resizedImg, cv.COLOR_BGR2GRAY)
# Detecting the faces with detectMultiScale()
faces = face_cascade.detectMultiScale(grayImg, 1.1, 4)
# Draw rectangle around faces
for (x, y, w, h) in faces:
    cv.rectangle(resizedImg, (x,y), (x + w, y + h), (255, 0, 0), 2) 

# Printing detected faces with squares
cv.imshow('img', resizedImg)
cv.waitKey(0) 

