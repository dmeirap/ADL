
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
        print('decoded', d.type, 'symbol', '"%s"' % d.data) 
        
        topLeftCorners = (d.rect.left, d.rect.top)
        topRightCorners = (d.rect.left + d.rect.width, d.rect.top) 
        bottomLeftCorners = (d.rect.left, d.rect.top + d.rect.height)     
        bottomRightCorners  = (d.rect.left + d.rect.width, d.rect.top + d.rect.height)  
        
        cv2.line(im, topLeftCorners, topRightCorners, (0, 255, 0), 2)
        cv2.line(im, topLeftCorners, bottomLeftCorners, (0, 255, 0), 2)
        cv2.line(im, topRightCorners, bottomRightCorners, (0, 255, 0), 2)
        cv2.line(im, bottomLeftCorners, bottomRightCorners, (0, 255, 0), 2)

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
        
        #************ SET PARAMETROS DE CAMARA *********************
        
        camera_matrix = np.array(
            [[781.45492812, 0, 932.52867315],
             [0,  782.46840213, 575.57313795],
             [0, 0, 1]], dtype="double"
        )
        
        dist_coeffs = np.zeros((5, 1),dtype="double")  # Lens distortion
        dist_coeffs[0,0] = 0.006589358509533379
        dist_coeffs[1,0] = -0.060555647700596946
        dist_coeffs[2,0] = -0.004607629905733067
        dist_coeffs[3,0] = -0.0035826292149932857
        dist_coeffs[4,0] = 0.038915448056452626
    
        try:
            (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_SQPNP)
        except:
            continue

        print("Rotation Vector:\n {0}".format(rotation_vector))
        print("Translation Vector:\n {0}".format(translation_vector))

        # Project a 3D point (0, 0, 1000.0) onto the image plane. Define la distancia estimada a la que debe estar la camara del objeto
        (center_end_point2D, jacobian) = cv2.projectPoints(np.array([(0.0, 0.0, 1500.0)]), rotation_vector, translation_vector, camera_matrix, dist_coeffs)

        for p in image_points:
            cv2.circle(im, (int(p[0]), int(p[1])), 3, (0, 0, 255), -1)

        p1 = (int(image_points[0][0]), int(image_points[0][1]))
        p2 = (int(center_end_point2D[0][0][0]), int(center_end_point2D[0][0][1]))

        # pillars
        cv2.line(im, bottomLeftCorners, p2, (255, 0, 0), 2)
        cv2.line(im, topLeftCorners, p2, (255, 0, 0), 2)
        cv2.line(im, bottomRightCorners, p2, (255, 0, 0), 2)
        cv2.line(im, topRightCorners, p2, (255, 0, 0), 2)
        
    cv2.imshow("Output", im)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cv2.destroyWindow("Output")