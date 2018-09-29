from __future__ import print_function
import numpy as np
import sys
import os
import cv2
caffe_root = '/home/work/glenn/gitmodel/caffe-ssd/'
sys.path.insert(0, caffe_root + 'python')  
import caffe  

import matplotlib
#%matplotlib inline
import pylab as plt
from matplotlib.pyplot import savefig
plt.switch_backend('agg')

testimg_dir='../check_img/'
result_dir='../test_result/'

'''
protofile='deploy.prototxt'
modelfile='snapshot_iter_40000_first.caffemodel'
outputlayer='Mconv7_stage4_new'

protofile='deploy.prototxt'            # titanx 22ms 
modelfile='snapshot_iter_42000_second.caffemodel'
outputlayer='Mconv7_stage4_new'
'''
protofile='deploy_3stage.prototxt'     # titanx 15-16ms 
modelfile='snapshot_iter_1000_third.caffemodel'
outputlayer='Mconv7_stage3_new'


def  drawPoints(im,point):
	cv2.circle(im,(point[0],point[1]),2,(0,255,0))
	cv2.circle(im,(point[2],point[3]),2,(0,255,0))
	cv2.circle(im,(point[4],point[5]),2,(0,255,0))
	cv2.circle(im,(point[6],point[7]),2,(0,255,0))
	cv2.circle(im,(point[8],point[9]),2,(0,255,0))
	return im
	
def filename(dir):
    filelist=[]
    for path,dirs,files in os.walk(dir):
	    filelist.append(files)
    return filelist

def eval():
    nh, nw = 112, 112
    img_mean = np.array([127.5, 127.5, 127.5], dtype=np.float32)
    filelist=filename(testimg_dir)
    print (len(filelist),filelist[0])
    #sys.exit()
    caffe.set_mode_gpu()
    caffe.set_device(0)
    proto=protofile
    model=modelfile
    net = caffe.Net(proto, model, caffe.TEST)

    
    for imagename in filelist[0]:
        im = caffe.io.load_image(testimg_dir+imagename)
        h, w, _ = im.shape
        if h < w:
            off = (w - h) / 2
            im = im[:, off:off + h]
        else:
            off = (h - w) / 2
            im = im[off:off + h, :]
        im = caffe.io.resize_image(im, [nh, nw])

        transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
        transformer.set_transpose('data', (2, 0, 1))  # row to col
        transformer.set_channel_swap('data', (2, 1, 0))  # RGB to BGR
        transformer.set_raw_scale('data', 255)  # [0,1] to [0,255]
        transformer.set_mean('data', img_mean)
        transformer.set_input_scale('data', 0.007843)

        net.blobs['data'].reshape(1, 3, nh, nw)
        net.blobs['data'].data[...] = transformer.preprocess('data', im)
        t1=cv2.getTickCount()
        out = net.forward()  
        t=(cv2.getTickCount()-t1)/cv2.getTickFrequency()
        print (t)
        pred = out[outputlayer]
        print (pred.shape,pred[0].shape,pred[0][0].shape)
        #print(pred[0][0])       
		 
        img=cv2.imread(testimg_dir+imagename)		
        pred[0]=pred[0]*255.0
		
        
        predimg=np.zeros((5,112,112))
        for i in range(5):
            predimg[i]=cv2.resize(pred[0][i],(112,112),interpolation=cv2.INTER_CUBIC)
        predimg=predimg.transpose(1,2,0)
		
        fig,ax=plt.subplots(1,5)
		
        for i in range(5):
            
            ax[i].imshow(img[:,:,[2,1,0]])
            ax1=ax[i].imshow(predimg[:,:,i],alpha=.5)
            fig = matplotlib.pyplot.gcf()
            cax = matplotlib.pyplot.gca()
            fig.set_size_inches(20, 7)
            fig.subplots_adjust(right=0.93)
            cbar_ax = fig.add_axes([0.95, 0.15, 0.01, 0.7])
            _ = fig.colorbar(ax1, cax=cbar_ax)
        savefig(result_dir+imagename[:-4]+'_'+str(i+5)+'_result.jpg')
        plt.close()
		
        fig,ax=plt.subplots(1,1)
        for i in range(5):
            ax2=ax.imshow(predimg[:,:,i],alpha=1.0/(i+1))
        fig = matplotlib.pyplot.gcf()
        cax = matplotlib.pyplot.gca()
        fig.set_size_inches(20, 20)
        fig.subplots_adjust(right=0.93)
        cbar_ax = fig.add_axes([0.95, 0.15, 0.01, 0.7])
        _ = fig.colorbar(ax2, cax=cbar_ax)
        savefig(result_dir+imagename[:-4]+'result_heatmaps.jpg')
        plt.close()
		
        for i in range(5):
            predimg_bi=cv2.resize(pred[0][i],(112,112),interpolation=cv2.INTER_CUBIC)
            maxvalue=np.max(predimg_bi)	
            re=np.where(predimg_bi==maxvalue)			
            cv2.imwrite(result_dir+imagename[:-4]+'_'+str(i)+'_result.jpg',predimg_bi)
            predimg_bi=cv2.imread(result_dir+imagename[:-4]+'_'+str(i)+'_result.jpg')
            if maxvalue>127:
                cv2.circle(predimg_bi,(re[1],re[0]),2,(255,0,0))
            cv2.imwrite(result_dir+imagename[:-4]+'_'+str(i)+'_result.jpg',predimg_bi)
            #img=cv2.resize(img,(112,112))
            #img=cv2.addWeighted(img,1,predimg_bi,0.3,0)#img+0.3*predimg
            #img=cv2.resize(img,(224,224),interpolation=cv2.INTER_CUBIC)
            #cv2.imwrite(result_dir+imagename[:-4]+'result.jpg',img)

    return


if __name__ == '__main__':
    eval()
