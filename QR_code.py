
from PIL import Image, ImageColor,ImageDraw, ImageFont 
import cv2 
from pyzbar.pyzbar import decode 
import numpy as np 

cap = cv2.VideoCapture(1) # 0 = (camara interna) ; 1 = (camara externa)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
cv2.namedWindow("Output", 0)

while True:
    ret, im = cap.read()
    if not ret:
        continue
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY, dstCn=0)
    
    for d in decode(im):
        
        print('\n\nDecoded: ', d.type, 'Symbol: ', '"%s"' % d.data) 
        
        topLeftCorners = d.polygon[1]
        topRightCorners = d.polygon[2] 
        bottomLeftCorners = d.polygon[0]     
        bottomRightCorners  = d.polygon[3] 
        
        im = cv2.polylines(im, [np.int32([d.polygon])], True, (0, 255, 0), 6)

        image_points = np.array([
            (int((topLeftCorners[0]+topRightCorners[0])/2),
             int((topLeftCorners[1]+bottomLeftCorners[1])/2)),     # Nose
            topLeftCorners,   
            topRightCorners,     
            bottomLeftCorners,     
            bottomRightCorners      
        ], dtype="double")
        
        # Coordenadas de cada esquina
        model_points = np.array([
            (0.0, 0.0, 0.0),             # Nose
            # Left top corner
            (-46.50, 46.50, 0),
            # Right top corner
            (46.50, 46.50, 0),
            # Left bottom corner
            (-46.50, -46.50, 0),
            # Right bottom corner
            (46.50, -46.50, 0)
        ],dtype="double")
        
        #************ SET PARAMETROS CAMARA *********************
        camera_matrix = np.array(
            [[773.74013464, 0, 957.40574841],
             [0,  774.15094158, 560.68397322],
             [0, 0, 1]], dtype="double"
        )
        
        dist_coeffs = np.zeros((5, 1),dtype="double")  # Lens distortion
        dist_coeffs[0,0] = -0.005905527127455908
        dist_coeffs[1,0] = 0.03188619089891427
        dist_coeffs[2,0] = -0.008733485736477005
        dist_coeffs[3,0] = -0.0007170139797843391
        dist_coeffs[4,0] = -0.06968794979128276
    
        try:
            (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_SQPNP)
        except:
            continue

        rmat, jac = cv2.Rodrigues(rotation_vector)
        angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)
        translacion = np.matrix(rmat).T * np.matrix(translation_vector)
        
        #print("\nRotation Pitch: {0}".format(angles[0]))
        print("\nRotation Yaw: {0}".format(angles[1]))
        #print("Rotation Roll: {0}".format(angles[2]))
        #print("\nTranslation X: {0}".format(translation_vector[0]))
        #print("Translation Y: {0}".format(translation_vector[1]))
        #print("\nTranslation Z:{0}".format(translation_vector[2]))
        print("\nTranslation:{0}".format(translacion[2]))
        
 
                
    cv2.imshow("Output", im)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cv2.destroyWindow("Output")