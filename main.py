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
# Clear all previously saved images
for file in os.listdir('./images/query'):
    os.remove('./images/query/'+file)
for file in os.listdir('./images/gallery'):
    os.remove('./images/gallery/'+file)
    
# Create color list
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

def main(queue,camID,initial_id,r_id,cam_id,unique_id):
    '''
    Detect, Track and save infomation of persons

    Objectives: 

    1. Use YOLO to detect person and store coordinates 
    2. Use DeepSORT to track detected persons throughout video frames
    3. Save detected persons for re-identification
    4. If re-identified, replace camID with global camID across camera views
    '''
    # Init YOLO model and load to memory
    yolo = YOLO()
    start = time.time()

    counter = []

    #deep_sort
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric)
    model_filename = 'model_data/market1501.pb'
    encoder = gdet.create_box_encoder(model_filename,batch_size=1)
    w = int(650)
    h = int(576)
    writeVideo_flag = True # Set to False to not save videos and detections
    if writeVideo_flag:
    # Define the codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        out = cv2.VideoWriter('./output/'+str(camID)+'_output.avi', fourcc, 10, (w, h))
        list_file = open('logs/detection_camera'+str(camID)+'.txt', 'w')
        frame_index = -1
    
    fps = 0.0

    frame_counter = 0

    #Diretory Creation
    if not os.path.isdir('./images/frames/'+str(camID)):
        os.mkdir('./images/frames/'+str(camID))
    if not os.path.isdir('./images/detections/'+str(camID)):
        os.mkdir('./images/detections/'+str(camID))

    # Empty folders
    for file in (sorted(os.listdir('./images/detections/'+str(camID)))):
        os.remove('./images/detections/'+str(camID)+'/'+file)
    for file in (sorted(os.listdir('./images/frames/'+str(camID)))):
        os.remove('./images/frames/'+str(camID)+'/'+file)
    

    while not queue.empty():
        # Retrieve a frame from the queue
        frame = queue.get()
        cv2.imwrite('./images/frames/'+str(camID)+'/'+str(frame_counter)+'.jpg',frame)
        frame_counter+=1
        t1 = time.time()
        # frame_copy --> to be cropped according to detected person and saved
        # frame_save --> Frame to be saved with video only showing unique camID
        frame_copy = frame.copy()
        frame_save = frame.copy()

        image = Image.fromarray(frame[...,::-1]) #bgr to rgb
        # Perform YOLO detection (Objective 1)
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

        # Call the tracker and update with current detections
        tracker.predict()
        tracker.update(detections)

        i = int(0)
        indexIDs = []
        boxes = []
        for det in detections:
            bbox = det.to_tlbr()
            #cv2.rectangle(frame,(int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(255,255,255), 2)
        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            #boxes.append([track[0], track[1], track[2], track[3]])
            # Store tracking_id as seperate variable for replacement
            tracking_id = track.track_id
            indexIDs.append(int(tracking_id))
            counter.append(int(track.track_id))
            bbox = track.to_tlbr()
            if int(bbox[0])<0 or int(bbox[1])<0 or int(bbox[2])<0 or int(bbox[3])<0:
                continue

            # Change camID's of re-identifed targets
            # Current using Camera camID 1 as the base of reference to compare other cameras
            if not camID == 1:
                for a in range(len(initial_id)):
                    if int(track.track_id) == initial_id[a]:
                        tracking_id = int(r_id[a]) # r_id is for only CAM 1
                        #Prevent donated camID from CAM 1 conflict with CAM 2 issued camID (which is coincidentally 1 as well)
                    elif int(track.track_id) == r_id[a]: #Prevent identital camID on 1 source
                        tracking_id = int(initial_id[a])
                    else:
                        tracking_id = track.track_id 
            else:
                tracking_id = track.track_id #Deepsort id
            color = [int(c) for c in COLORS[tracking_id]]
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(color), 3) #bbox[0] and [1] is startpoint [2] [3] is endpoint

            # Select which camID to be displayed (Local or Global if re-identified)
            display_id = tracking_id
            for b in range(len(unique_id)):
                if tracking_id == r_id[b]:
                    display_id = unique_id[b]
                    
            cv2.putText(frame,str(display_id),(int(bbox[0]), int(bbox[1] -10)),0, 5e-3 * 150, (color),2)

            if len(class_names) > 0:
               class_name = class_names[0]
               #cv2.putText(frame, str(class_names[0]),(int(bbox[0]), int(bbox[1] -20)),0, 5e-3 * 150, (color),2)
               
            # Save bounding box data for re-identification (Objective 3)
               frame1 = frame_copy[int(bbox[1]):int(bbox[3]),int(bbox[0]):int(bbox[2])]#create instance of cropped frame using current frame, crop according to bounding box coordinates
               query_path = image_path+'/query'
               gallery_path = image_path+'/gallery'
            
               #Perform resizing to ensure equal size of features extracted from images
               frame2 = cv2.resize(frame1,(46,133),interpolation = cv2.INTER_AREA) #resize cropped image
               cv2.imwrite('./images/detections/'+str(camID)+'/frame'+str(frame_counter)+'_'+str(tracking_id)+'.jpg',frame2)


               if not camID == 1:
                   dst_path = gallery_path
                   #if file does not exist --> save
                   file_path = dst_path+'/'+str(tracking_id)+'_'+str(camID)+'.png' 
                   if frame_counter % 10 == 0 or not os.path.isfile(file_path):
                       cv2.imwrite(file_path,frame2)#save cropped frame


               if camID == 1:    
                   dst_path = query_path 
                    #if file does not exist --> save
                   file_path = dst_path+'/'+str(tracking_id)+'.png' 
                   if frame_counter % 10 == 0 or not os.path.isfile(file_path):
                        cv2.imwrite(file_path,frame2)#save cropped frame

            # Draw bounding boxes and Unique IDs for video to be saved
            if tracking_id in initial_id or tracking_id in r_id:
                
                if tracking_id in initial_id and not camID == 1:
                    index = initial_id.index(tracking_id)
                    color = [int(c) for c in COLORS[int(unique_id[index].split('P')[1])]] #Determine color of bounding box
                    # Draw box and label with unique ID
                    cv2.rectangle(frame_save, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(color), 3) #bbox[0] and [1] is startpoint [2] [3] is endpoint
                    cv2.putText(frame_save,str(unique_id[index]),(int(bbox[0]), int(bbox[1] -10)),0, 5e-3 * 150, (color),2)

                elif tracking_id in r_id and camID == 1:
                    index = r_id.index(tracking_id)
                    color = [int(c) for c in COLORS[int(unique_id[index].split('P')[1])]]
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
        # Visualize result

        cv2.putText(frame, "FPS: %f"%(fps),(int(20), int(40)),0, 5e-3 * 200, (0,255,0),3)
        cv2.namedWindow('Camera '+str(camID), 0)
        cv2.resizeWindow('Camera '+str(camID), w ,h)
        cv2.imshow('Camera '+str(camID), frame)

        if writeVideo_flag:
            #save a frame
            frame_save = cv2.resize(frame_save,(w,h)) # Resize frame to fit video
            out.write(frame_save)
            frame_index = frame_index + 1
            # Write detections onto file
            list_file.write('./images/frames/'+str(camID)+'/'+str(frame_counter)+'.jpg'+' | ')
            if len(boxs) != 0:
                for i in range(0,len(boxs)):
                    list_file.write(str(boxs[i][0]) + ' '+str(boxs[i][1]) + ' '+str(boxs[i][2]) + ' '+str(boxs[i][3]) + ' ' + str(class_names[i][0])+', ')
            list_file.write('\n')

        fps  = ( fps + (1./(time.time()-t1)) ) / 2 

        # Press Q to stop!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    print(" ")
    print("[Finish]")
    end = time.time()

    if len(pts[track.track_id]) != None:
       print(source_names[camID-1]+": "+ str(count) + " " + str(class_name) +' Found')
       
    else:
       print("[No Found]")

    if writeVideo_flag:
        out.release()
        list_file.close()
    
    print('Time taken: '+str(round(end-start))+' seconds')

    cv2.destroyAllWindows()

def start_queue(q,source):
    '''
    Initlize VideoCapture with source and retrieves frames from video

    Objectives:
    - Read video source and store frame into Queue 
    '''
    video_capture = cv2.VideoCapture(source,cv2.CAP_FFMPEG)
    while True:
        ret,frame = video_capture.read() # Read video source
        if not ret:
            break
        q.put(frame) # Put frame into Queue
    video_capture.release()
    print("capture complete")    

#Fixed feature extractor for cameras
def extract_query(initial_id,r_id,cam_id,flag,unique_id):
    '''
    Extract features and perform cosine similarity to dertimine re-identification

    Objective:
    1. Read images extracted from YOLO+DeepSORT and store features, afterwhich image will be deleted
    2. Compare features of all images 
    3. If feature match > 70%, store both IDs to be replaced in main tracker
    4. Store all initial re-identifications and updated ones onto text file
    5. Clear extracted features and repeat process

    '''
    reid_full = open('logs/reid_full.txt','w')
    start = time.time()
    
    while not flag.is_set():
        time.sleep(5)
        #reset all stored info when called again (prevents continous stacking) (Objective 5)
        query_features.clear()
        q_id.clear()
        gallery_features.clear()
        g_id.clear()
        cam_num.clear()
        query_list = sorted(os.listdir('./images/query'))
        gallery_list = sorted(os.listdir('./images/gallery'))

        #Read and Extract features from images (Objective 1)
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

        # Comparsion of features (Objective 2)
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
                            source_names
                # Store matched ID in inital_id for replacing (Objective 3)
                if not highest_score == 0:
                    #Update
                    if int(g_id[gallery_num]) in initial_id: #If initial camID is already in list
                        index = initial_id.index(int(g_id[gallery_num])) #Get index of camID stored
                        if not r_id[index] == int(q_id[query_num]): #Ensure that the 2 targets are not previously defined as a match
                            r_id[index] = int(q_id[query_num]) #Update value
                            print('ID '+g_id[gallery_num]+' updated to '+q_id[query_num])
                            # (Objective 4)
                            reid_full.write('ID '+g_id[gallery_num]+' updated to '+q_id[query_num]+' at '+str(round(time.time()-start))+' seconds')
                            reid_full.write('\n')
                        else:
                            pass

                    #New creation
                    elif int(q_id[query_num]) not in r_id and int(g_id[gallery_num]) not in initial_id and not (g_id[gallery_num] == q_id[query_num]):
                        initial_id.append(int(g_id[gallery_num]))
                        #Create and append CAM 1 id into r_id which r_id is a global list in the Manager
                        #r_id contains matches with CAM 1
                        r_id.append(int(q_id[query_num]))
                        unique_id.append(unique_prefix+str(len(initial_id)))
                        print(q_id[query_num] +' identified with '+g_id[gallery_num]+' on camera '+cam_num[gallery_num])
                        # (Objective 4)
                        reid_full.write(q_id[query_num] +' identified with '+g_id[gallery_num]+' on camera '+cam_num[gallery_num]+' at '+str(round(time.time()-start))+' seconds')
                        reid_full.write('\n')
    reid_full.close()

            

if __name__ == '__main__':
    '''
    Process flow:
    1. Declare Processes to run
    2. Run in order: Video frame in queue, detect
    3. If at least 2 sources inputted, start re-id Process

    4. At end of process, write re-identified targets onto reid.txt
    '''
    processes = int(1) #Initialize the first camera (in the arguments) wuth camID 1
    data = args["input"].split(',') # list of cameras input
    process_capture = [] #Store Processes to run
    process_read = []

    #CL-> Explain how each of the different list works together 
    '''
    As Multiprocessing creates different processes, the CPU will allocate a different sector of memory to each process
    With each having a different sector of memory, sharing of data in that situation is not possible
    With use of Managers, it creates a global shared memory between processes which all processes are able to have reference to
    '''
    manager = Manager()
    initial_id = manager.list() # Store Initial camID before replacement
    r_id = manager.list() # Store camID to be replaced to (with reference to camera 1)
    cam_id = manager.list() # Store camera camID of detected re-identifications
    unique_id = manager.list() # Store list of Unique IDs
    # Flag to indicate that all main detections processes have terminated, and indicate re-id process to end. 
    # Else it'll be a zombie running in background
    flag = Event() 

    rp = Process(target=extract_query,args=(initial_id,r_id,cam_id,flag,unique_id))
 
    #Run up reid process only when more than 1 camera source is inputted
    if len(data) > 1: # Only run re-identification if more than 1 camera source declared (Obj 3)
        rp.start()
    
    #Declare Queue objects and Processes (Obj 1)
    #For each camera, two separate process will be forked to handle queue and detection
    #"Process capture" : Process to handle the ingestion of the camera frames into buffer
    #"Process read" : Process to handle the retrieval of the buffer and then perform detection
    for i in range(len(data)):
        queue_name = 'processes_'+str(processes)
        queue_name = manager.Queue()
        # Queue first, then detection (Obj 2)
        # Process for Queueing, to send over name of Queue and data source 
        p = Process(target=start_queue,args=(queue_name,data[i]))
        # Process for detection, to send over Queue name, amount of processes, initial_id and r_id lists for re-identification, 
        # detected cam_id and unique_ids list
        p1 = Process(target=main,args=(queue_name,processes,initial_id,r_id,cam_id,unique_id))
        process_capture.append(p)

        process_read.append(p1)
        processes+=1 #Assign a new camera camID (increment) for the next camera source to be read
    #Start Processses
    for p in process_capture:
        p.start()
    time.sleep(1)# Delay for frames to be read first as the reading of the RTSP streaming will be slower.
    for p in process_read:
        p.start()
 
    #To handle the "zombie" state where the parent (the main program) process terminate while the child process is still alive.
    #By joining the two processes, the termination of one will lead to the termination of the other.
    for p in process_read:
        p.join() # Join parent process to child process (For parent process to wait for child process to end before terminating)

    #In event when more than 1 camera is available, the reid will be run, thus this is to end the reid process.    
    if rp.is_alive():
        flag.set() # Set flag to indicate re-identification process to end

        # Write re-identifications onto text file with unique IDs
        rid_file = open('logs/reid.txt', 'w')
        for i in range(len(initial_id)):
            rid_file.write("ID "+str(initial_id[i]) +' reidentified as '+str(r_id[i]))
            rid_file.write('\n')
            rid_file.write("Uniquely identified as "+unique_id[i])
            rid_file.write('\n\n')
        rid_file.close()

    while rp.is_alive(): # Wait for re-id process to end
        pass

    print('all processes completed')

