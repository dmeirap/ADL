from PIL import Image, ImageColor,ImageDraw, ImageFont 
import cv2 
from pyzbar.pyzbar import decode, ZBarSymbol
import numpy as np 
import time
import openpyxl
import desviacion

def media(angulo):
    listaYawprint = list()
    for i in range(3):
        number = angulo
        listaYawprint.append(number)
        
    sumatoria = float(sum(listaYawprint))
    
            
cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
cv2.namedWindow("Output", 0)

while True:
    ret, im = cap.read()
    if not ret:
        continue
    
    for d in decode(im, symbols=[ZBarSymbol.QRCODE]):
        
        print('\n\nCodigo: ', d.type, 'Symbol: ', d.data) 
        
        topLeftCorners = d.polygon[1]
        topRightCorners = d.polygon[2] 
        bottomLeftCorners = d.polygon[0]     
        bottomRightCorners  = d.polygon[3]
        
        im = cv2.polylines(im, [np.array([d.polygon])], True, (255, 255, 0), 6)
        im = cv2.circle(im, d.polygon[1], radius=10, color=(0, 0, 255),thickness=10)
        im = cv2.circle(im, d.polygon[2], radius=10, color=(255, 0, 0),thickness=10)
        im = cv2.circle(im, d.polygon[0], radius=10, color=(0, 255, 0),thickness=10)
        im = cv2.circle(im, d.polygon[3], radius=10, color=(0, 0, 0),thickness=10)
        
        #Esquinas del QR
        image_points = np.array([
            topLeftCorners,   
            topRightCorners,
            bottomRightCorners,     
            bottomLeftCorners    
        ], dtype="double")
        
        # Dimensiones QR
        model_points = np.array([
            (-46.50, 46.50, 0),
            (46.50, 46.50, 0),
            (46.50, -46.50, 0),
            (-46.50, -46.50, 0)
        ],dtype="double")
        
        #******* SET PARAMETROS CAMARA **********
        camera_matrix = np.array(
            [[776.34474311, 0, 946.84727768],
            [0,  779.03968654, 550.44881788],
            [0, 0, 1]], dtype="double"
        )
        
        dist_coeffs = np.array(
                    [[0.00975813],
                     [-0.02868024],
                     [-0.01228495],
                     [-0.00277628],
                     [-0.00719628]], dtype="double"
        )
    
        try:
            (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs)
        except:
            continue
        
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
        angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)
        translacion = np.matrix(rmat).T * np.matrix(translation_vector)
        control_yaw = rmat[0][1]
        control_roll = rmat[2][0]
            
        if control_yaw > 0.5 and control_roll > 0:
            print("\nRotation Yaw: ",angles[0])
            print("Distancia Z: ",translacion.item(2))
                        
        elif control_yaw < 0.5 and control_roll > 0:
            print("\nRotation Yaw: ",-(angles[1]))
            print("Distancia Z: ",translacion.item(2))           
        
                        
    cv2.imshow("Output", im)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cv2.destroyWindow("Output")