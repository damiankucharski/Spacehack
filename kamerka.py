import numpy as np
import cv2
from tensorflow import keras
import cvlib as cv
from cvlib.object_detection import draw_bbox
model = keras.models.load_model(r'C:\Users\d.kucharski\Documents\Python Scripts\Spacehacks\Spacehack\model.h5')
url = 'http://100.111.30.77:8080/video'
cap = cv2.VideoCapture(0)
 
font                   = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (10,20)
fontScale              = 1
fontColor              = (255,255,255)
lineType               = 2
 
 
while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    resized = cv2.resize(frame, (300, 300))
    # Our operations on the frame come here
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    bbox, labels, conf = cv.detect_common_objects(resized)
    for i in range(len(labels)):
        bottomLeftCornerOfText = (int(bbox[i][0]), int(bbox[i][1]))
        cv2.putText(resized, labels[i], 
            bottomLeftCornerOfText, 
            font, 
            fontScale,
            fontColor,
            lineType)
 
 
    result = model(np.expand_dims(resized, axis = 0))
    result = np.float32(result)
 
 
    # Using cv2.putText() method 
    # image = cv2.putText(image, 'OpenCV', org, font,  
    #                 fontScale, color, ) 
 
    bbox, labels, conf = cv.detect_common_objects(frame)
    for i in range(len(labels)):
        cv2.putText(np.float32(result),'Hello World!', 
            bottomLeftCornerOfText, 
            font, 
            fontScale,
            fontColor,
            lineType)
    #result = draw_bbox(result, bbox, label, conf)
    #print([[label_i, bbox_i] for label_i, bbox_i in zip(labels,bbox)])
 
    # Display the resulting frame
    result = np.array(result)
    print(result.shape)
    cv2.imshow('frame',np.squeeze(result))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
 
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()