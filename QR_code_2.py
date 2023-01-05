
from PIL import Image, ImageColor,ImageDraw, ImageFont #pip install Pillow
import cv2 #pip install opencv-python
from pyzbar.pyzbar import decode #pip install pyzbar
import numpy as np #pip install numpy

cap = cv2.VideoCapture(0) #Capturar un video frame by frame
cv2.namedWindow("Output", 0)
while True:
    ret, im = cap.read()
    if not ret:
        continue
    size = im.shape 
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY, dstCn=0)
        
    for d_1 in decode(im):
        
        for d_2 in decode(im):
            
            if d_1.data != d_2.data:
                print('----- DOS CODIGOS DETECTADOS -----')
                print('Decoded', d_1.type, 'Symbol', '"%s"' % d_1.data)
                print('Decoded', d_2.type, 'Symbol', '"%s"' % d_2.data)
                topLeftCorners_1 = (d_1.rect.left, d_1.rect.top)
                topRightCorners_1 = (d_1.rect.left + d_1.rect.width, d_1.rect.top) 
                bottomLeftCorners_1= (d_1.rect.left, d_1.rect.top + d_1.rect.height)     
                bottomRightCorners_1  = (d_1.rect.left + d_1.rect.width, d_1.rect.top + d_1.rect.height)
                topLeftCorners_2 = (d_2.rect.left, d_2.rect.top)
                topRightCorners_2 = (d_2.rect.left + d_2.rect.width, d_2.rect.top) 
                bottomLeftCorners_2= (d_2.rect.left, d_2.rect.top + d_2.rect.height)     
                bottomRightCorners_2  = (d_2.rect.left + d_2.rect.width, d_2.rect.top + d_2.rect.height)  
                
                cv2.line(im, topLeftCorners_1, topRightCorners_1, (255, 0, 0), 2)
                cv2.line(im, topLeftCorners_1, bottomLeftCorners_1, (255, 0, 0), 2)
                cv2.line(im, topRightCorners_1, bottomRightCorners_1, (255, 0, 0), 2)
                cv2.line(im, bottomLeftCorners_1, bottomRightCorners_1, (255, 0, 0), 2)
                cv2.line(im, topLeftCorners_2, topRightCorners_2, (0, 255, 0), 2)
                cv2.line(im, topLeftCorners_2, bottomLeftCorners_2, (0, 255, 0), 2)
                cv2.line(im, topRightCorners_2, bottomRightCorners_2, (0, 255, 0), 2)
                cv2.line(im, bottomLeftCorners_2, bottomRightCorners_2, (0, 255, 0), 2)

                image_points_1 = np.array([
                    (int((topLeftCorners_1[0]+topRightCorners_1[0])/2),
                    int((topLeftCorners_1[1]+bottomLeftCorners_1[1])/2)),     # Nose
                    topLeftCorners_1,     # Left eye left corner
                    topRightCorners_1,     # Right eye right corne
                    bottomLeftCorners_1,     # Left Mouth cornerq
                    bottomRightCorners_1      # Right mouth corner
                ], dtype="double")
                
                image_points_2 = np.array([
                    (int((topLeftCorners_2[0]+topRightCorners_2[0])/2),
                    int((topLeftCorners_2[1]+bottomLeftCorners_2[1])/2)),     # Nose
                    topLeftCorners_2,     # Left eye left corner
                    topRightCorners_2,     # Right eye right corne
                    bottomLeftCorners_2,     # Left Mouth cornerq
                    bottomRightCorners_2      # Right mouth corner
                ], dtype="double")
                
                # 3D model points.
                model_points = np.array([
                    (0.0, 0.0, 0.0),             # Nose
                    # Left eye left corner
                    (-225.0, 170.0, -135.0),
                    # Right eye right corner
                    (225.0, 170.0, -135.0),
                    # Left Mouth corner
                    (-150.0, -150.0, -125.0),
                    # Right mouth corner
                    (150.0, -150.0, -125.0)

                ],dtype="double")
                
                # Camera internals
                focal_length = size[1]
                center = (size[1]/2, size[0]/2)
                camera_matrix = np.array(
                    [[focal_length, 0, center[0]],
                    [0, focal_length, center[1]],
                    [0, 0, 1]], dtype="double"
                )

                print("Camera Matrix :\n {0}".format(camera_matrix))
                
                dist_coeffs = np.zeros((4, 1),dtype="double")  # Assuming no lens distortion
                
                try:
                    (success, rotation_vector_1, translation_vector_1) = cv2.solvePnP(model_points, image_points_1, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_SQPNP)
                    (success, rotation_vector_2, translation_vector_2) = cv2.solvePnP(model_points, image_points_2, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_SQPNP)
                except:
                    continue

                print("Rotation Vector 1:\n {0}".format(rotation_vector_1))
                print("Rotation Vector 2:\n {0}".format(rotation_vector_2))
                print("Translation Vector 1:\n {0}".format(translation_vector_1))
                print("Translation Vector 2:\n {0}".format(translation_vector_2))

                # Project a 3D point (0, 0, 1000.0) onto the image plane. Define la distancia estimada a la que debe estar la camara del objeto
                # We use this to draw a line sticking out of the nose

                (nose_end_point2D_1, jacobian) = cv2.projectPoints(np.array([(0.0, 0.0, 1500.0)]), rotation_vector_1, translation_vector_1, camera_matrix, dist_coeffs)
                (nose_end_point2D_2, jacobian) = cv2.projectPoints(np.array([(0.0, 0.0, 1500.0)]), rotation_vector_2, translation_vector_2, camera_matrix, dist_coeffs)
                
                for p_1 in image_points_1:
                    cv2.circle(im, (int(p_1[0]), int(p_1[1])), 3, (0, 0, 255), -1)
                
                for p_2 in image_points_2:
                    cv2.circle(im, (int(p_2[0]), int(p_2[1])), 3, (0, 0, 255), -1)
                
                p1_1 = (int(image_points_1[0][0]), int(image_points_1[0][1]))
                p1_2 = (int(nose_end_point2D_1[0][0][0]), int(nose_end_point2D_1[0][0][1]))
                p2_1 = (int(image_points_2[0][0]), int(image_points_2[0][1]))
                p2_2 = (int(nose_end_point2D_2[0][0][0]), int(nose_end_point2D_2[0][0][1]))

                # pillars
                cv2.line(im, bottomLeftCorners_1, p1_2, (255, 0, 0), 2)
                cv2.line(im, topLeftCorners_1, p1_2, (255, 0, 0), 2)
                cv2.line(im, bottomRightCorners_1, p1_2, (255, 0, 0), 2)
                cv2.line(im, topRightCorners_1, p1_2, (255, 0, 0), 2)
                cv2.line(im, bottomLeftCorners_2, p2_2, (0, 255, 0), 2)
                cv2.line(im, topLeftCorners_2, p2_2, (0, 255, 0), 2)
                cv2.line(im, bottomRightCorners_2, p2_2, (0, 255, 0), 2)
                cv2.line(im, topRightCorners_2, p2_2, (0, 255, 0), 2)
            
    for d in decode(im):  
        
        print('----- UN CODIGOS DETECTADOS -----')
        print('Decoded', d.type, 'Symbol', '"%s"' % d.data) 
        
        topLeftCorners = (d.rect.left, d.rect.top)
        topRightCorners = (d.rect.left + d.rect.width, d.rect.top) 
        bottomLeftCorners = (d.rect.left, d.rect.top + d.rect.height)     
        bottomRightCorners  = (d.rect.left + d.rect.width, d.rect.top + d.rect.height)  
        
        cv2.line(im, topLeftCorners, topRightCorners, (255, 0, 0), 2)
        cv2.line(im, topLeftCorners, bottomLeftCorners, (255, 0, 0), 2)
        cv2.line(im, topRightCorners, bottomRightCorners, (255, 0, 0), 2)
        cv2.line(im, bottomLeftCorners, bottomRightCorners, (255, 0, 0), 2)
        
        image_points = np.array([
            (int((topLeftCorners[0]+topRightCorners[0])/2),
             int((topLeftCorners[1]+bottomLeftCorners[1])/2)),     # Nose
            topLeftCorners,     # Left eye left corner
            topRightCorners,     # Right eye right corne
            bottomLeftCorners,     # Left Mouth cornerq
            bottomRightCorners      # Right mouth corner
        ], dtype="double")

        # 3D model points.
        model_points = np.array([
            (0.0, 0.0, 0.0),             # Nose
            # Left eye left corner
            (-225.0, 170.0, -135.0),
            # Right eye right corner
            (225.0, 170.0, -135.0),
            # Left Mouth corner
            (-150.0, -150.0, -125.0),
            # Right mouth corner
            (150.0, -150.0, -125.0)

        ],dtype="double")
        
        # Camera internals
        focal_length = size[1]
        center = (size[1]/2, size[0]/2)
        camera_matrix = np.array(
            [[focal_length, 0, center[0]],
             [0, focal_length, center[1]],
             [0, 0, 1]], dtype="double"
        )

        print("Camera Matrix :\n {0}".format(camera_matrix))
        
        dist_coeffs = np.zeros((4, 1),dtype="double")  # Assuming no lens distortion
        
        try:
            (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_SQPNP)
        except:
            continue

        print("Rotation Vector:\n {0}".format(rotation_vector))
        print("Translation Vector:\n {0}".format(translation_vector))

        # Project a 3D point (0, 0, 1000.0) onto the image plane. Define la distancia estimada a la que debe estar la camara del objeto
        # We use this to draw a line sticking out of the nose

        (nose_end_point2D, jacobian) = cv2.projectPoints(np.array([(0.0, 0.0, 1000.0)]), rotation_vector, translation_vector, camera_matrix, dist_coeffs)

        for p in image_points:
            cv2.circle(im, (int(p[0]), int(p[1])), 3, (0, 0, 255), -1)

        p1 = (int(image_points[0][0]), int(image_points[0][1]))
        p2 = (int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))

        # pillars
        cv2.line(im, bottomLeftCorners, p2, (255, 0, 0), 2)
        cv2.line(im, topLeftCorners, p2, (255, 0, 0), 2)
        cv2.line(im, bottomRightCorners, p2, (255, 0, 0), 2)
        cv2.line(im, topRightCorners, p2, (255, 0, 0), 2)
    
                            
        # Display image
    cv2.imshow("Output", im)

    # Wait for the magic key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.destroyWindow("Output")