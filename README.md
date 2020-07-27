# Multi Person Tracking

1. Install requirements

```
pip3 install -r requirements.txt 
```
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
  

OUTPUTS:

Images extracted in ./images

Re-Identification between camera views is based on Cosine Similarity of features, as declared in extract_query()
