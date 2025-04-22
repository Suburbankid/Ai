import cv2
import matplotlib.pyplot as plt

filename =input("entern person name")
dataset_path="./data/"
cam=cv2.VideoCapture(0)
model=cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")
cropped_face =None
while True:
    success, img = cam.read()
    if not success:
        print("reading Camera Failed")
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces=model.detectMultiScale(img,1.3,5)
    faces=sorted(faces,key=lambda f:f[2]*f[3])
    if len(faces)>0:
        f=faces[-1]
        x,y,w,h = f
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
        cropped_face = img[y:y+h,x:x+w]
    cv2.imshow("Image",img)
    cv2.imshow("cropp",cropped_face)
    key = cv2.waitKey(1)
    if key ==ord('q'):
        break


cam.release()
cv2.destroyAllWindows()




