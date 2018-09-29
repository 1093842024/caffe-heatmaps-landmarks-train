import numpy as np
import cv2
import h5py
import sys


def createheatmap(landmark_x,landmark_y):
    heatmap=np.zeros((28,28),dtype=np.float)
    x=landmark_x*28
    y=landmark_y*28
    for i in range(28):
	    for j in range(28):
		    if (x-i)*(x-i)+(y-j)*(y-j)<=4:
			    heatmap[j][i]=1.0/(1+(x-i)*(x-i)*2+(y-j)*(y-j)*2)
    return heatmap
			
        

	
def createheatmaps(landmarks):
    heatmaps=[]
    for i in range(5):
        heatmap=createheatmap(landmarks[2*i],landmarks[2*i+1])
        heatmaps.append(heatmap)
        #cv2.imwrite('test_heatmap_'+str(i)+'.png',heatmap)
    #return np.concatenate((heatmaps[0], heatmaps[1],heatmaps[2],heatmaps[3],heatmaps[4]), axis = 1)  #(112,560)
    return np.array(heatmaps)



#heatmap=createheatmap(0.532,0.356)
#cv2.imwrite('test_heatmap.png',heatmap*255)


landmarks=[0.532,0.356,0.800,0.329,0.637,0.576,0.631,0.651,0.875,0.614]
heatmaps=createheatmaps(landmarks)
print heatmaps.shape
print heatmaps[0]
print heatmaps[1]
print heatmaps[2]
print heatmaps[3]
print heatmaps[4]




with open('landmarks_face.txt', 'r') as f:
    lines=f.readlines()
imglist=[]
landmarkslist=[]
for line in lines:
    line=line.strip().split(' ')
    imglist.append(line[0])
    landmarkslist.append(line[1:])

f.close()

heatmaps=[]
landmarkslist=np.array(landmarkslist, float)
print landmarkslist.shape

for landmarks in landmarkslist:
    heatmaps.append(createheatmaps(landmarks))

print len(heatmaps)

train_file_path = 'train_heatmaps.hd5'
with h5py.File(train_file_path, 'w') as f1:
    f1['heatmaps'] = heatmaps
	

imgs=open('imglist.txt','w')
for line in imglist:
    imgs.write(line+'\n')
imgs.close()
	

	
	
	
	
