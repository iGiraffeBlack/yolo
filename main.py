#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function, absolute_import
import os
import datetime
from timeit import time
import warnings
import sys
warnings.filterwarnings('ignore')
from cv2 import cv2
import numpy as np
import argparse
import glob
from multiprocessing import Queue, Process, Manager, Event
from multiprocessing.managers import BaseManager
from PIL import Image
from yolo import YOLO
from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
from deep_sort.detection import Detection as ddet
from collections import deque
from keras import backend

import lomo
import json
from scipy import spatial

#Load LOMO Configuration
with open('lomo_config.json','r') as f:
    lomo_config = json.load(f)

backend.clear_session()
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input",help="path to input videos, multiple seperated by ','", default = "rtsp://192.168.1.136:8554/stream")
ap.add_argument("-c", "--class",help="name of class", default = "person")
args = vars(ap.parse_args())

pts = [deque(maxlen=30) for _ in range(9999)]
warnings.filterwarnings('ignore')

os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;udp" #set protocol for rtsp

# initialize a list of colors to represent each possible class label
np.random.seed(100)

image_path = './images'
if not os.path.isdir(image_path):
    os.mkdir(image_path)

#create query and gallery folders
if not os.path.isdir(image_path+'/query'):
    os.mkdir(image_path+'/query')
if not os.path.isdir(image_path+'/gallery'):
    os.mkdir(image_path+'/gallery')

for file in os.listdir('./images/query'):
    os.remove('./images/query/'+file)
for file in os.listdir('./images/gallery'):
    os.remove('./images/gallery/'+file)

COLORS = np.random.randint(0, 255, size=(200, 3),
    dtype="uint8")

source_names = args["input"].split(',')
#Definition of the parameters
max_cosine_distance = 0.5#0.9 余弦距离的控制阈值
nn_budget = None #Size of Feature representation (if None, no limit is used)
nms_max_overlap = 0.3 #非极大抑制的阈值

q_id = []#to store id of extracted person from camera 1
g_id = []

query_features = []#store features
gallery_features = []

cam_num = []

unique_prefix = 'P'

class ymanager(BaseManager):
    pass

ymanager.register('YOLO',YOLO)

def main(yolo,queue,ID,initial_id,r_id,cam_id,unique_id):
    #yolo = YOLO()
    start = time.time()

    counter = []
    #deep_sort
    
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric)
    model_filename = 'model_data/market1501.pb'
    encoder = gdet.create_box_encoder(model_filename,batch_size=1)
    writeVideo_flag = True
    if writeVideo_flag:
    # Define the codec and create VideoWriter object
        w = int(650)
        h = int(576)
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        out = cv2.VideoWriter('./output/'+str(ID)+'_output.avi', fourcc, 10, (w, h))
    
    fps = 0.0

    frame_counter = 0

    while not queue.empty():
        frame = queue.get()
        frame_counter+=1
        t1 = time.time()
        frame_copy = frame.copy()
        frame_save = frame.copy()

        image = Image.fromarray(frame[...,::-1]) #bgr to rgb
        boxs,class_names = yolo.detect_image(image)
        backend.clear_session()
        features = encoder(frame,boxs)
        # score to 1.0 here).
        detections = [Detection(bbox, 1.0, feature) for bbox, feature in zip(boxs, features)]
        # Run non-maxima suppression.
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        indices = preprocessing.non_max_suppression(boxes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]

        # Call the tracker
        tracker.predict()
        tracker.update(detections)

        i = int(0)
        indexIDs = []
        c = []
        boxes = []
        for det in detections:
            bbox = det.to_tlbr()
            #cv2.rectangle(frame,(int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(255,255,255), 2) #remove 2 bounding boxes on same person
        #remove all current images
        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            #boxes.append([track[0], track[1], track[2], track[3]])
            tracking_id = track.track_id
            indexIDs.append(int(tracking_id))
            counter.append(int(track.track_id))
            bbox = track.to_tlbr()
            if int(bbox[0])<0 or int(bbox[1])<0 or int(bbox[2])<0 or int(bbox[3])<0:
                continue
            
            if not ID == 1:
                for a in range(len(initial_id)):
                    if int(track.track_id) == initial_id[a]:
                        if ID == cam_id[a]:
                            tracking_id = int(r_id[a])
                    elif int(track.track_id) == r_id[a]: #Prevent identital ID on 1 source
                        if ID == cam_id[a]:
                            tracking_id = int(initial_id[a])
                    else:
                        tracking_id = track.track_id 
            else:
                tracking_id = track.track_id
            color = [int(c) for c in COLORS[tracking_id]]
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(color), 3) #bbox[0] and [1] is startpoint [2] [3] is endpoint
            cv2.putText(frame,str(tracking_id),(int(bbox[0]), int(bbox[1] -10)),0, 5e-3 * 150, (color),2)
            if len(class_names) > 0:
               class_name = class_names[0]
               #cv2.putText(frame, str(class_names[0]),(int(bbox[0]), int(bbox[1] -20)),0, 5e-3 * 150, (color),2)
               
            #save bounding box data
               frame1 = frame_copy[int(bbox[1]):int(bbox[3]),int(bbox[0]):int(bbox[2])]#create instance of cropped frame using current frame, crop according to bounding box coordinates
               query_path = image_path+'/query'
               gallery_path = image_path+'/gallery'
               if not os.path.isdir(query_path):
                   os.mkdir(query_path)
               if not os.path.isdir(gallery_path):
                   os.mkdir(gallery_path)
               frame2 = cv2.resize(frame1,(46,133),interpolation = cv2.INTER_AREA) #resize cropped image
               if not ID == 1:
                   dst_path = gallery_path
                   #if file does not exist --> save
                   file_path = dst_path+'/'+str(tracking_id)+'_'+str(ID)+'.png' 
                   if frame_counter % 10 == 0 or not os.path.isfile(file_path):
                       cv2.imwrite(file_path,frame2)#save cropped frame

               if ID == 1:    
                   dst_path = query_path 
                    #if file does not exist --> save
                   file_path = dst_path+'/'+str(tracking_id)+'.png' 
                   if frame_counter % 10 == 0 or not os.path.isfile(file_path):
                        cv2.imwrite(file_path,frame2)#save cropped frame
            

            if ID == 1:
                if tracking_id in r_id:
                    index = r_id.index(tracking_id)
                    cv2.rectangle(frame_save, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(color), 3) #bbox[0] and [1] is startpoint [2] [3] is endpoint
                    cv2.putText(frame_save,str(unique_id[index]),(int(bbox[0]), int(bbox[1] -10)),0, 5e-3 * 150, (color),2)
            elif ID != 1:
                if tracking_id in initial_id:
                    index = initial_id.index(tracking_id)
                    cv2.rectangle(frame_save, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(color), 3) #bbox[0] and [1] is startpoint [2] [3] is endpoint
                    cv2.putText(frame_save,str(unique_id[index]),(int(bbox[0]), int(bbox[1] -10)),0, 5e-3 * 150, (color),2)


            i += 1
            #bbox_center_point(x,y)
            center = (int(((bbox[0])+(bbox[2]))/2),int(((bbox[1])+(bbox[3]))/2))
            #track_id[center]
            pts[track.track_id].append(center)
            #center point
            #cv2.circle(frame,  (center), 1, color, 5)
            '''
	    #draw motion path
            for j in range(1, len(pts[track.track_id])):
                if pts[track.track_id][j - 1] is None or pts[track.track_id][j] is None:
                   continue
                thickness = int(np.sqrt(64 / float(j + 1)) * 2)
                cv2.line(frame,(pts[track.track_id][j-1]), (pts[track.track_id][j]),(color),thickness)
                #cv2.putText(frame, str(class_names[j]),(int(bbox[0]), int(bbox[1] -20)),0, 5e-3 * 150, (255,255,255),2)
            '''
        count = len(set(counter))
        
        #cv2.putText(frame, "Total Object Counter: "+str(count),(int(20), int(120)),0, 5e-3 * 200, (0,255,0),2)
        #cv2.putText(frame, "Current Object Counter: "+str(i),(int(20), int(80)),0, 5e-3 * 200, (0,255,0),2)
        cv2.putText(frame, "FPS: %f"%(fps),(int(20), int(40)),0, 5e-3 * 200, (0,255,0),3)
        cv2.namedWindow('Camera '+str(ID), 0)
        cv2.resizeWindow('Camera '+str(ID), 650 ,576)
        cv2.imshow('Camera '+str(ID), frame)

        if writeVideo_flag:
            #save a frame
            frame_save = cv2.resize(frame_save,(650,576))
            out.write(frame_save)

        fps  = ( fps + (1./(time.time()-t1)) ) / 2 
        #print(set(counter))

        # Press Q to stop!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        if queue.empty():
            break
    print(" ")
    print("[Finish]")
    end = time.time()

    if len(pts[track.track_id]) != None:
       print(source_names[ID-1]+": "+ str(count) + " " + str(class_name) +' Found')
       
    else:
       print("[No Found]")

    if writeVideo_flag:
        out.release()
    
    print('Time taken: '+str(round(end-start))+' seconds')

    cv2.destroyAllWindows()

def start_queue(q,source):
    video_capture = cv2.VideoCapture(source,cv2.CAP_FFMPEG)
    while True:
        ret,frame = video_capture.read()
        if not ret:
            break
        q.put(frame)
    video_capture.release()
    print("capture complete")    

#Fixed feature extractor for cameras
def extract_query(initial_id,r_id,cam_id,flag,unique_id):
    while not flag.is_set():
        time.sleep(5)
        #reset all stored info when called again (prevents continous stacking)
        query_features.clear()
        q_id.clear()
        gallery_features.clear()
        g_id.clear()
        query_list = sorted(os.listdir('./images/query'))
        gallery_list = sorted(os.listdir('./images/gallery'))
        #Extract features from images
        for file in query_list:
            image = cv2.imread(os.path.join('./images/query',file))
            result = lomo.LOMO(image,lomo_config)
            query_features.append(result)#Append to list
            q_id.append(file.split('.')[0])
            os.remove('./images/query/'+file)#Remove image after features extracted
            
        for file in gallery_list:
            image = cv2.imread(os.path.join('./images/gallery',file))
            camera_id = file.split('_')[1].split('.')[0]
            result = lomo.LOMO(image,lomo_config)
            gallery_features.append(result)
            g_id.append(file.split('_')[0])
            cam_num.append(camera_id)#Append camera number
            os.remove('./images/gallery/'+file)
        for cam in range(1,len(source_names)):
            for i in range(len(query_features)):
                highest_score = 0
                for j in range(len(gallery_features)):
                    if not int(cam_num[j]) == cam+1:
                        continue
                    cos_sim = 1 - spatial.distance.cosine(query_features[i],gallery_features[j])

                    if cos_sim > 0.7:
                        if cos_sim > highest_score:
                            highest_score = cos_sim
                            query_num = i
                            gallery_num = j
                            

                if not highest_score == 0:
                    if int(g_id[gallery_num]) in initial_id: #If initial ID is already in list
                        index = initial_id.index(int(g_id[gallery_num])) #Get index of ID stored
                        if not r_id[index] == int(q_id[query_num]) and cam_id[index] == int(cam_num[gallery_num]):
                            r_id[index] = int(q_id[query_num]) #Update value
                            print('ID '+g_id[gallery_num]+' updated to '+q_id[query_num])
                        else:
                            pass

                    elif int(q_id[query_num]) not in r_id and int(g_id[gallery_num]) not in initial_id and not (g_id[gallery_num] == q_id[query_num]):
                        initial_id.append(int(g_id[gallery_num]))
                        r_id.append(int(q_id[query_num]))
                        cam_id.append(int(cam_num[gallery_num]))
                        unique_id.append(unique_prefix+str(len(initial_id)))
                        print(q_id[query_num] +' identified with '+g_id[gallery_num]+' on camera '+cam_num[gallery_num])
            

if __name__ == '__main__':
    processes = int(1) 
    data = args["input"].split(',')
    process_capture = [] #Store Processes to run
    process_read = []
    manager = Manager()
    mymanager = ymanager()
    mymanager.start()
    yolo = mymanager.YOLO()
    initial_id = manager.list()
    r_id = manager.list()
    cam_id = manager.list()
    unique_id = manager.list()
    flag = Event()
    rp = Process(target=extract_query,args=(initial_id,r_id,cam_id,flag,unique_id))
    if len(data) > 1:
        rp.start()
    
    #Declare Queue objects and Processes
    for i in range(len(data)):
        queue_name = 'processes_'+str(processes)
        queue_name = manager.Queue()
        p = Process(target=start_queue,args=(queue_name,data[i]))
        p1 = Process(target=main,args=(yolo,queue_name,processes,initial_id,r_id,cam_id,unique_id))
        process_capture.append(p)
        process_read.append(p1)
        processes+=1
    #Start Processses
    for p in process_capture:
        p.start()
    time.sleep(2)#Delay for frames to be read first
    for p in process_read:
        p.start()
    for p in process_read:
        p.join()
    if rp.is_alive():
        flag.set()
        rid_file = open('reid.txt', 'w')
        for i in range(len(initial_id)):
            rid_file.write("ID "+str(initial_id[i]) +' reidentified as '+str(r_id[i]))
            rid_file.write('\n')
            rid_file.write("Uniquely identified as "+unique_id[i])
            rid_file.write('\n\n')
        rid_file.close()
    while rp.is_alive():
        pass

    print('all processes completed')

