import matplotlib.pyplot as plt
import cv2
import numpy as np
from sklearn.cluster import KMeans
img=cv2.imread("noah.jpg")
print(img.shape)
img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
img=cv2.resize(img,(382,306))
plt.imshow(img)
plt.show()
x=img.reshape((-1,3))
print(x.shape)


k=4
model=KMeans(n_clusters=k)
model.fit(x)
centroids=model.cluster_centers_
print(centroids)
colors=np.array(centroids,dtype='uint8')
print(colors)

i=1
for color in colors:
    plt.subplot(1,k,i)
    plt.axis("off")
    i=i+1
    mat=np.zeros((100,100,3),dtype='uint8')
    mat[:,:,:]=color
    plt.imshow(mat)
plt.show()
