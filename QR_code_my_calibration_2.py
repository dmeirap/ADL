import zbar
from PIL import Image, ImageColor
import cv2
import numpy as np 
import math
import time
import collections
from matplotlib import pyplot as plt
            
cap = cv2.VideoCapture("videos/Video_Prueba_Dron_1200_config1.mp4")
#cap = cv2.VideoCapture(2)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
#cap.set(cv2.CAP_PROP_BRIGHTNESS, -64)
#cap.set(cv2.CAP_PROP_CONTRAST, 5)
#cap.set(cv2.CAP_PROP_SATURATION, 0)
#cap.set(cv2.CAP_PROP_GAMMA, 80)
#cap.set(cv2.CAP_PROP_SHARPNESS, 0)
scanner = zbar.ImageScanner()
scanner.parse_config('enable')
cv2.namedWindow("Output", 0)

pre_timeframe=0
new_timeframe=0
yaw_values=collections.deque(maxlen=50)
pitch_values=collections.deque(maxlen=50)
dist_values=collections.deque(maxlen=50)

while True:
  
    ret, im = cap.read()
    
    if not ret:
        continue
    
    new_timeframe=time.time()
    fps=1/(new_timeframe-pre_timeframe)
    pre_timeframe=new_timeframe
    fps=int(fps)
    cv2.putText(im, str(fps), (1775,1050), cv2.FONT_HERSHEY_SIMPLEX,3,(100,255,0),4)
    cv2.putText(im, "Yaw: " ,(10,50), cv2.FONT_HERSHEY_SIMPLEX,1,(100,255,0),3)
    cv2.putText(im, "Distancia QR: ", (10,100), cv2.FONT_HERSHEY_SIMPLEX,1,(100,255,0),3)

    size = im.shape
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY, dstCn=0)
    pil = Image.fromarray(gray)
    width, height = pil.size
    raw = pil.tobytes()
    image = zbar.Image(width, height, 'Y800', raw)
    scanner.scan(image)
    
    # Dimensiones QR
    model_points = np.array([
            (0.0, 0.0, 0.0),
            (-46.50, 46.50, 0),
            (46.50, 46.50, 0),
            (46.50, -46.50, 0),
            (-46.50, -46.50, 0),
            (0, 46.50, 0),
            (46.50, 0, 0),
            (0, -46.50, 0)
    ],dtype="float32")
        
    #******* SET PARAMETROS CAMARA **********
    camera_matrix = np.array(
            [[1.51187075e+03, 0.00000000e+00, 9.73619124e+02],
            [0.00000000e+00, 1.51847952e+03, 3.55830121e+02],
            [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]], dtype="float32"
    )
        
    dist_coeffs = np.array(
                    [[-2.33996748e-02],
                    [7.68975435e-01],
                    [-9.51846221e-03],
                    [1.27593135e-03],
                    [-1.79462157e+00]], dtype="float32"
    )    

    for d in image:
        
        print('\n\nCodigo: ', d.type, 'Symbol: ', d.data) 
        
        
        try:
            topLeftCorners, bottomLeftCorners, bottomRightCorners, topRightCorners  = [item for item in d.location]
        except:
            continue
      
        cv2.line(im, topLeftCorners, topRightCorners, (255, 255, 0), 3)
        cv2.line(im, topLeftCorners, bottomLeftCorners, (255, 255, 0), 3)
        cv2.line(im, topRightCorners, bottomRightCorners, (255, 255, 0), 3)
        cv2.line(im, bottomLeftCorners, bottomRightCorners, (255, 255, 0), 3)
        
        #Esquinas del QR
        image_points = np.array([
            (int((topLeftCorners[0]+topRightCorners[0])/2.0),int((topLeftCorners[1]+bottomLeftCorners[1])/2.0)),   
            topLeftCorners,    #bottomLeft 
            topRightCorners, #bottomRight
            bottomRightCorners,    #topRight
            bottomLeftCorners,     
            ((topLeftCorners[0]+topRightCorners[0])/2.0,(topLeftCorners[1]+topRightCorners[1])/2.0), 
            ((topRightCorners[0]+bottomRightCorners[0])/2.0,(topRightCorners[1]+bottomRightCorners[1])/2.0), 
            ((bottomLeftCorners[0]+bottomRightCorners[0])/2.0,(bottomLeftCorners[1]+bottomRightCorners[1])/2.0)
        ], dtype="float32")
        

        #criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 0.001)
        #grey_frame = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        #cv2.cornerSubPix(grey_frame,image_points,(11,11),(-1,-1),criteria)

	
        try:
      
            (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs, 8)
            print("\n",rotation_vector)
            
        except:
            continue
        if not success:
            print("NOT SUCCESS")
        #*********EJES DE COORDENADAS*************
        unitv_points = np.array([[0,0,0], [46.50,0,0], [0,46.50,0], [0,0,46.50]], dtype = 'float32').reshape((4,1,3))
        axis_points, jac = cv2.projectPoints(unitv_points, rotation_vector, translation_vector, camera_matrix, dist_coeffs)        
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0,0,0)]
        if len(axis_points) > 0:
                axis_points = axis_points.reshape((4,2))
                origin = (int(axis_points[0][0]),int(axis_points[0][1]) )
                for p, c in zip(axis_points[1:], colors[:3]):
                    p = (int(p[0]), int(p[1]))
                    if origin[0] > 5*im.shape[1] or origin[1] > 5*im.shape[1]:break
                    if p[0] > 5*im.shape[1] or p[1] > 5*im.shape[1]:break
                    cv2.line(im, origin, p, c, 5)
                        
        rmat, jac = cv2.Rodrigues(rotation_vector)
        angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat.T)
        translacion = -np.matrix(rmat).T * np.matrix(translation_vector)

        if  rotation_vector[0] > 0 and -0.25 < rotation_vector[2] < 0.25 and 15 < (180 + angles[0]) < 20:  
            yaw_values.append(angles[1])
            dist_values.append(translacion.item(2))
            #pitch_values.append(angles[0])
            print("\nRotation Pitch: ",180  + angles[0])
            print("\nRotation Yaw: ",angles[1])
            print("\nDistancia Z: ",translacion.item(2))
            #plt.clf()
            plt.figure("Yaw")
            plt.clf()
            plt.plot(yaw_values)
            #plt.figure("Distancia Z")
            #plt.clf()
            #plt.plot(dist_values)
            plt.pause(0.01)
            cv2.putText(im, str(angles[1]), (90,50), cv2.FONT_HERSHEY_SIMPLEX,1,(100,255,0),3)
            cv2.putText(im, str(translacion.item(2)), (220,100), cv2.FONT_HERSHEY_SIMPLEX,1,(100,255,0),3)
        
        
    cv2.imshow("Output", im) 
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
    	
    	break
    	
 
cv2.destroyWindow("Output")
