from json import load
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import imutils
import time
import cv2
import os

def detect_and_predict_mask(frame, facenet, maskNet):
    #Grabbing the frame from the input given using camera and converting it into blob (Binary Large OBjects)
    (h,w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (224,224),
    (104.0,177.0,123.0))

    #Passing the blob throough the faceNet to obtain ROI i.e our face.
    facenet.setInput(blob)
    detections = facenet.forward()
    print(detections.shape)

    faces=[]
    locs=[]
    preds=[]

    #Looping over the detections
    for i in range(0, detections.shape[2]):
    
        #Finding probability associated with the current detection
        confidence = detections[0,0,i,2]
        
        #If probability > 0.5 i.e "Masked"
        if confidence > 0.5:
        
            #Calculating the coordinates for the bounding box.
            box = detections[0,0,i,3:7] * np.array([w,h,w,h])
            (startX, startY, endX, endY) = box.astype("int")

            #Checking if the coordinates lie within the range of frame
            (startX, startY) = (max(0, startX), max(0,startY))
            (endX, endY) = (min(w-1, endX), min(h-1, endY))

            #Extracting Face ROI and convering it into RGB as OpenCV takes input as BGR but our model takes input as RGB
            face = frame[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224,224))
            face = img_to_array(face)
            face = preprocess_input(face)

            #Appending data of Bounding box.
            faces.append(face)
            locs.append((startX, startY, endX, endY))
    #If atleast one face is found in the frame.
    if len(faces) > 0:
        
        #Detecting face and making prediction if mask is on or not.
        faces = np.array(faces, dtype="float32")
        preds = maskNet.predict(faces, batch_size=32)
        
    return (locs,preds)

#Importing pre-trained face detection model
prototxtPath = r"face_detector\deploy.prototxt"
weightsPath = r"face_detector\res10_300x300_ssd_iter_140000.caffemodel"
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

#Loading our Mask Detector Model
maskNet = load_model("mask_detector.model")

print("STARTING VIDEO STREAM...")
vs = VideoStream(src=0).start()


#Looping over frames in the VideoStream
while True:

    frame = vs.read()
    frame = imutils.resize(frame, width=400)

    #Calling function to detect if face in current frame is wearing a mask or not. 
    (locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)

    #Looping over locations and prediction
    for(box, pred) in zip(locs, preds):
        #unpacking the locations and assigning the locs coordinates as box points.
        (startX, startY, endX, endY) = box
        (mask, withoutMask) = pred

        #Giving label and color for Output
        label = "Mask" if mask > withoutMask else "No Mask"
        color = (0,255,0) if label=="Mask" else (0,0,255)

        label = "{}: {:.2f}%".format(label, max(mask, withoutMask)*100)
        
        #Displating the bounding box and label over the face
        cv2.putText(frame, label, (startX, startY - 10), 
        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        cv2.rectangle(frame, (startX, startY), (endX, endY), color ,2)

    #Show the output frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    
    #Setting up quiting key.
    if key == ord("q"):
        break
cv2.destroyAllWindows()
vs.stop()