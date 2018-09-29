# -*- coding: UTF-8 -*-
import numpy as np
import cv2
import os
import numpy.random as npr
import random

def IoU(box, boxes):
    """Compute IoU between detect box and gt boxes

    Parameters:
    ----------
    box: numpy array , shape (5, ): x1, y1, x2, y2, score
        input box
    boxes: numpy array, shape (n, 4): x1, y1, x2, y2
        input ground truth boxes

    Returns:
    -------
    ovr: numpy.array, shape (n, )
        IoU
    """
    box_area = (box[2] - box[0] + 1) * (box[3] - box[1] + 1)
    area = (boxes[:, 2] - boxes[:, 0] + 1) * (boxes[:, 3] - boxes[:, 1] + 1)
    xx1 = np.maximum(box[0], boxes[:, 0])
    yy1 = np.maximum(box[1], boxes[:, 1])
    xx2 = np.minimum(box[2], boxes[:, 2])
    yy2 = np.minimum(box[3], boxes[:, 3])

    # compute the width and height of the bounding box
    w = np.maximum(0, xx2 - xx1 + 1)
    h = np.maximum(0, yy2 - yy1 + 1)

    inter = w * h
    ovr = inter*1.0 / (box_area + area - inter)
    return ovr

def  drawPoints(im,point):
	cv2.circle(im,(point[0],point[1]),2,(0,255,0))
	cv2.circle(im,(point[2],point[3]),2,(0,255,0))
	cv2.circle(im,(point[4],point[5]),2,(0,255,0))
	cv2.circle(im,(point[6],point[7]),2,(0,255,0))
	cv2.circle(im,(point[8],point[9]),2,(0,255,0))
	return im
	
def seperatebox(pts,scale):
	x1=float(pts[0])
	y1=float(pts[1])
	x2=float(pts[2])
	y2=float(pts[3])
	x3=float(pts[8])
	y3=float(pts[9])
	x4=float(pts[6])
	y4=float(pts[7])
	x_center=(x1+x2+x3+x4)/4
	y_center=(y1+y2+y3+y4)/4
	x11=x_center-(x_center-x1)*scale
	y11=y_center-(y_center-y1)*scale
	x22=x_center-(x_center-x2)*scale
	y22=y_center-(y_center-y2)*scale
	x33=x_center-(x_center-x3)*scale
	y33=y_center-(y_center-y3)*scale
	x44=x_center-(x_center-x4)*scale
	y44=y_center-(y_center-y4)*scale
	vertx=[x11,x22,x33,x44]
	verty=[y11,y22,y33,y44]
	return vertx,verty

def pnpoly(nvert,vertx,verty,x,y):
	c=0
	i=0
	j=nvert-1
	while(i<nvert):
		if (verty[i]>y)!=(verty[j]>y) and (x<(vertx[j]-vertx[i]) * (y-verty[i]) / (verty[j]-verty[i]) + vertx[i]):
			c=~c
		j=i
		i+=1
	return c

def tfactor(img):
    hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV);#增加饱和度光照的噪声
    hsv[:,:,0] = hsv[:,:,0]*(0.8+ np.random.random()*0.2);
    hsv[:,:,1] = hsv[:,:,1]*(0.3+ np.random.random()*0.7);
    hsv[:,:,2] = hsv[:,:,2]*(0.2+ np.random.random()*0.8);
    img = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR);
    return img 
	
	
anno_file = "./CelebA/Anno/list_bbox_celeba_fix.txt"
anno_landmarks_file="./CelebA/Anno/list_landmarks_celeba_fix.txt"
im_dir = "./CelebA/Img/img_celeba.7z/img_celeba/"
save_dir = "landmarks_face"
bigangle_save_dir="bigangle_landmarks_face"

if not os.path.exists(bigangle_save_dir):
    os.mkdir(bigangle_save_dir)
if not os.path.exists(save_dir):
    os.mkdir(save_dir)
with open(anno_file, 'r') as f:
    annotations = f.readlines()
with open(anno_landmarks_file,'r') as fl:
    annotations_lands=fl.readlines()
num = len(annotations)
print "%d pics in total" % num

f1 = open('landmarks_face.txt', 'w')
f2=open('error_img.txt','w')
count=0

biganglesample=0
normalanglesample=0

for annotation,annotation_land in zip(annotations,annotations_lands):

    if count<2:
	    count+=1
	    continue
	
    count+=1	
    annotation = annotation.strip().split(' ')
    annotation_land=annotation_land.strip().split(' ')
    im_path = annotation[0]
    bbox = map(int, annotation[1:])
    landmarks=map(int,annotation_land[1:])
    img = cv2.imread(im_dir+im_path)
    width,height,channel=img.shape
    if (count-2)%500==0:
        print im_dir+im_path
    #img=drawPoints(img,landmarks)
    
    x1, y1, w, h = bbox
    x2=x1+w
    y2=y1+h
    box=np.array([x1,y1,x2,y2])
	
    # ignore small faces
    if min(w, h) < 35 or x1 < 0 or y1 < 0 :
	    print ('small face or out of range '+im_path)
	    f2.write('small face or out of range '+im_path+'\n')
	    continue
		
    bigangle=0
    vertx,verty=seperatebox(landmarks,0.75)
    c=pnpoly(len(vertx),vertx,verty,float(landmarks[4]),float(landmarks[5]))
    if c==0:
        bigangle=1
    for i in range (10):
	    size = npr.randint(int(min(w,h) * 0.7), np.ceil(1.3 * max(w,h)))
        # delta here is the offset of box center
	    delta_x = npr.randint(-w * 0.3, w * 0.3)
	    delta_y = npr.randint(-h * 0.3, h * 0.3)

	    nx1 = int(max(x1 + w / 2 + delta_x - size / 2, 0))
	    ny1 = int(max(y1 + h / 2 + delta_y - size / 2, 0))
	    nx2 = nx1 + size
	    ny2 = ny1 + size
	    if nx2 > width or ny2 > height:
		    continue
	    crop_box = np.array([nx1, ny1, nx2, ny2])
	    cropped_im = img[ny1 : ny2, nx1 : nx2, :]
	    resized_im = cv2.resize(cropped_im, (112,112), interpolation=cv2.INTER_LINEAR)
	
	    offset_x1=(landmarks[0]-nx1)*1.0/size
	    offset_y1=(landmarks[1]-ny1)*1.0/size
	    offset_x2=(landmarks[2]-nx1)*1.0/size
	    offset_y2=(landmarks[3]-ny1)*1.0/size
	    offset_x3=(landmarks[4]-nx1)*1.0/size
	    offset_y3=(landmarks[5]-ny1)*1.0/size
	    offset_x4=(landmarks[6]-nx1)*1.0/size
	    offset_y4=(landmarks[7]-ny1)*1.0/size
	    offset_x5=(landmarks[8]-nx1)*1.0/size
	    offset_y5=(landmarks[9]-ny1)*1.0/size
	    box_ = box.reshape(1, -1)
	    if IoU(crop_box, box_) >= 0.6:
		    if bigangle==1:
		        save_file = os.path.join(bigangle_save_dir, str(i)+'_'+im_path)
		        biganglesample+=1
		    else:
		        if biganglesample>=normalanglesample:
				    normalanglesample+=1
				    save_file= os.path.join(save_dir, str(i)+'_'+im_path)
		        else:
				    break
		    f1.write( save_file+ ' %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f\n'%(offset_x1, offset_y1,offset_x2, offset_y2,offset_x3, offset_y3,offset_x4, offset_y4,offset_x5, offset_y5))
		    if random.choice([0,1]) > 0:   #光照变化
			    resized_im=tfactor(resized_im)
		    cv2.imwrite(save_file, resized_im)
		    #break
		    #print (save_file)

f1.close()
f2.close()
	
	
	
	
	