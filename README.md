# caffe-heatmaps-landmarks-train
self-design an train stratege with caffe refering to openpose thought 

##create traindata and mask label file

>1)generate_landmarksface_celebA.py
>2)generate_maps.py
>3)create_lmdb.py

##train & deploy prototxt ,solver file

>1)train_3stage.prototxt
>2)deploy_3stage.prototxt
>3)solver.prototxt

##eval result

>eval_image.py

##result example
![example1](https://github.com/1093842024/caffe-heatmaps-landmarks-train/blob/master/eval_result/0_000184_9_result.jpg)
![example2](https://github.com/1093842024/caffe-heatmaps-landmarks-train/blob/master/eval_result/0_000245_9_result.jpg)
