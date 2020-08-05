# Multi Person Tracking

1. Install requirements

```
pip3 install -r requirements.txt 
```
Code above has been tested with Python 3.6.9, Tensorflow 1.14.0 and Keras 2.3.1 

Download model from Google drive and place into model_data folder
https://drive.google.com/file/d/1AKvHo_JnBgZk93-3Idyn8E__wolc3ZLh/view?usp=sharing

Or convert your own YOLO model 
```
python3 convert.py model_data/yolov3.cfg model_data/yolov3.weights model_data/yolo.h5
```
How to run file: 

1. Launch main.py with file as inputs, seperated by ','
```
python3 main.py -i file1.mp4,file2.mp4
```
If running with RTSP, 
```
~ -i rtsp://IP:PORT/STREAM NAME
```  

Sample videos are located at
https://drive.google.com/drive/folders/1uSFOzQ2dJggSoTdsmNCCAesD1IVIR-ND?usp=sharing
  

OUTPUTS:

Images extracted in ./images

Re-Identification between camera views is based on Cosine Similarity of features, as declared in extract_query()

HOW TO read detection files:

[path to frame] <bbox[0]> <bbox[1]> <bbox[2]> <bbox[3]> 

Frames are saved in images/frames

Detections are saved in images/detections

Github references:
#### [deep_sort_yolov3](https://github.com/yehengchen/Object-Detection-and-Tracking/tree/master/OneStage/yolo/deep_sort_yolov3)
#### [LOMO feature extractor](https://github.com/dongb5/LOMO-feature-extractor)
#### [DeepSORT](https://github.com/nwojke/deep_sort)
#### [YOLOv3](https://github.com/Qidian213/deep_sort_yolov3)

Others:
#### [Multiprocessing python](https://docs.python.org/3/library/multiprocessing.html)

Read for uses on Process, Manager, Event and Queues

#### [LOMO paper](https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Liao_Person_Re-Identification_by_2015_CVPR_paper.pdf)


TODO list:

Add re-identification for multiple cameras (more than 2)

Able to reference each ID to each other (Current: Cam 1 to all, New: Cam any to all)

Prevent hardcoding of video height and width

