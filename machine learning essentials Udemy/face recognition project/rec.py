import cv2
import numpy as np
import os
dataset_path="./data/"
facedata =[]
offset=20
labels=[]
map={}
classid=0
for f in os.listdir(dataset_path):
    if f.endswith(".npy"):
        map[classid]=f[:-4]
        dataitem=np.load(dataset_path +f)
        m=dataitem.shape[0]
        facedata.append(dataitem)
        target=classid*np.ones((m,))
        classid+=1
        labels.append(target)
Xt =np.concatenate(facedata,axis=0)
yt =np.concatenate(labels,axis=0).reshape((-1,1))
print(Xt.shape)
print(yt.shape)
print(map)
#algorith
def dist(p,q):
    return np.sqrt(np.sum((p-q)**2))

def knn(X,y,xt,k=5):
    m= X.shape[0]
    dlist =[]
    for i in range(m):
        d=dist(X[i],xt)
        dlist.append((d,y[i]))
    dlist=sorted(dlist)
    dlist=np.array(dlist[:k], dtype=object)
    labels=dlist[:,1]
    labels, cnts=np.unique(labels,return_counts=True)
    idx=cnts.argmax()
    pred=labels[idx]
    return int(pred)

#recooo
cam=cv2.VideoCapture(0)
model=cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")
while True:
    success, img = cam.read()
    if not success:
        print("reading Camera Failed")

    faces=model.detectMultiScale(img,1.3,5)

    for f in faces:
        x,y,w,h = f
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
        cropped_face = img[y-offset:y+h+offset,x-offset:x+w+offset]
        if cropped_face.shape[0]>0 and cropped_face.shape[1]>0:
        
            cropped_face=cv2.resize(cropped_face,(100,100))
            if cropped_face.size == Xt.shape[1]:
        
                classpredicted=knn(Xt,yt,cropped_face.flatten())
                namepredicted=map[classpredicted]
                cv2.putText(img, namepredicted,(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2,cv2.LINE_AA)
                cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)

    cv2.imshow("predctionwindow",img)
    key = cv2.waitKey(1)
    if key ==ord('q'):
        break


cam.release()
cv2.destroyAllWindows()
