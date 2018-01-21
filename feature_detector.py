import cv2
"""
Created By Matthew MacFarquhar
This script uses opencv-python to identify features of the human faces
the face is outlined in blue, the eyes are in green, the mouth is red
"""
#Load the haarcascade files obtained from github stored locally
eyes_cascade=cv2.CascadeClassifier("haarcascade_eye.xml")
face_cascade =cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
mouth_cascade =cv2.CascadeClassifier("haarcascade_smile.xml")

#indictaes if user has pressed th 'p' key. set to False to start
takePic = False

def checkForFeatures(gray_img, img):
    """
    This function takes in the gray_img and img params
    it finds faces in the gray_img and puts a rectangle around the face on the corresponding img \
    then passes the modifed image onto finding the checkForEyesAndMouth function
    """
    #create a list of faces using opencv
    faces= face_cascade.detectMultiScale(gray_img,scaleFactor=1.4, minNeighbors=6)

    #declare global boolean for if there is a face detected
    global faceVisible
    faceVisible=False

    #declare the border properties for the detected face
    left_face=None
    right_face=None
    bottom_face=None
    top_face=None

    #keep track of faces created
    faceCount = 0

    #the faces list is 4 tuples store as x top left y top left width and height
    for x,y,w,h in faces:
        #create the blue rectangle and initialize the border properties
        img=cv2.rectangle(img,(x,y),(x+w,y+h), (255,0,0), 2)
        left_face=x
        right_face=x+w
        bottom_face=y+h
        top_face=y

        #found a face
        faceCount = faceCount + 1

        #person only has 1 face so stop the loop
        if faceCount == 1:
            #say we found a face
            faceVisible=True
            break

    #call the eyes and mouth finder and give the face box dimensions
    return checkForEyesAndMouth(gray_img,img, left_face, right_face, bottom_face, top_face)
def checkForEyesAndMouth(gray_img, img, left_face,right_face,bottom_face,top_face):

    """
    This function finds the position of two eyes and draws them to the image
    then it passes the modified image along with the gray image and border params to the checkForMouth function
    """

    #creates list of eyes locations using opencv
    eyes = eyes_cascade.detectMultiScale(gray_img,scaleFactor=1.2,minNeighbors=6)

    #declares a global variable indicating if the camera has found 2 eyes
    global eyeVisable
    eyeVisable=False

    #create variables to hold the y value of the bottom of the eye and the number of eyes drawn
    bottom_eye = 0
    eyeCount =0;

    #loop through the tuples (x,y,width,height) in eyes
    for x,y,w,h in eyes:
        #create a green rectangle around the eye, sets y value of the bottom of the eye
        img=cv2.rectangle(img,(x,y),(x+w,y+h), (0,255,0), 2)
        bottom_eye= y+h

        #increment number of eyes drawn
        eyeCount=eyeCount+1

        #when there are 2 eyes, stop searching for additional eyes. set eyeVisable to true because 2 eyes are found
        if eyeCount ==2:
            eyeVisable=True
            break

    #pass the gray_img, image, bottom of eye position, and face borders to the find mouth function
    return checkForMouth(gray_img,img,bottom_eye, right_face,left_face,bottom_face, top_face)
def checkForMouth(gray_img,img,bottom_eye, right_face, left_face, bottom_face, top_face):
    """
    This function uses the positioning parameters and the images to find a mouth.
    then draws a red rectangle around it.
    """
    #create a list of the locations of mouths
    mouths = mouth_cascade.detectMultiScale(gray_img,scaleFactor=1.3,minNeighbors=3)

    #create a var for if there is a mouth detected
    global mouthVisable
    mouthVisable= False

    #make a mouth counter and set to 0
    mouthCount = 0

    #loop through the tuples (x,y,width,height) of the mouths list
    for x,y,w,h in mouths:
        #if the top of the mouth is higher than the bottom of the eye or if no face was found or if the mouth does not fall
        #within any of the bounderies, go to the next mouth.
        #the smile haarcascade is very finicky so I needed to put in some limmiters to where the position can be
        if y < bottom_eye or bottom_face==None or y+h > bottom_face or x < left_face or x+w > right_face or y < top_face:
            continue

        #create a red rectangle around the mouth
        img=cv2.rectangle(img,(x,y),(x+w,y+h), (0,0,255), 2)

        #incrememt mouths. if there is a mouth then say we found a mouth and stop searching
        mouthCount = mouthCount + 1
        if mouthCount == 1:
            mouthVisable=True
            break

    #return the final image with all the rectangles drawn
    return img

#use the first camera on the device's camera list
video =cv2.VideoCapture(0)
while True:
    #store the img on camera in image, check is true if the camera is working
    check, img = video.read()

    #make a gray version
    gray_img=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #find the features and set it equal to the image
    img=checkForFeatures(gray_img,img)

    #show the image with the rectangles
    cv2.imshow("capturing (press q to quit, p to take a 'smart picture')",img)

    #wait 1 ms
    key = cv2.waitKey(1)

    #if the 'p' key was pressed and all features are presesnt close all windows and show the picture end the video
    if takePic and mouthVisable and eyeVisable and faceVisible:
        cv2.destroyAllWindows()
        cv2.imshow("face", img)
        cv2.waitKey(0)
        break

    #if the user presses 'p' say you want to take a picture when all features are presesnt
    if key ==ord('p'):
        takePic= True

    #if the user presses q end the video and show the last image
    if key ==ord('q'):
        cv2.destroyAllWindows()
        cv2.imshow("face",img)
        cv2.waitKey(0)
        break
#end all windows
cv2.destroyAllWindows()
