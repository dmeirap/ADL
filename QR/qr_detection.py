import cv2
import numpy as np 
import time 

cap = cv2.VideoCapture(1) 
qr = cv2.QRCodeDetector()
#cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
#cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
window_name = 'OpenCV QR Code'
cv2.namedWindow(window_name, 0)

while True:
    ret, im = cap.read()
     
    if ret:
        ret_qr, decoded_info, qr_points, _ = qr.detectAndDecodeMulti(im)
        if ret_qr:
            for s, p in zip(decoded_info, qr_points):
                
                if s:
                    print("\nInformacion QR: ", s)
                    color = (0, 255, 0)
                else:
                    color = (0, 0, 255)
                
                topLeftCorners = qr_points[0][3]
                topRightCorners = qr_points[0][2]
                bottomLeftCorners = qr_points[0][0]    
                bottomRightCorners  = qr_points[0][1]
                
                im = cv2.polylines(im, [p.astype(int)], True, color, 6)
                """ im = cv2.circle(im, bottomLeftCorners, radius=5, color=(0, 255, 0),thickness=10)
                im = cv2.circle(im, bottomRightCorners, radius=5, color=(0, 0, 255),thickness=10)
                im = cv2.circle(im, topRightCorners, radius=5, color=(255, 0, 0),thickness=10)
                im = cv2.circle(im, topLeftCorners, radius=5, color=(0, 0, 0),thickness=10) """
        
                #print(qr_points)
                
                image_points = np.array([
                    bottomLeftCorners,
                    bottomRightCorners,
                    topRightCorners,
                    topLeftCorners           
                ], dtype="double") 
                print(image_points)
                model_points = np.array([ 
                    (-46.50, 46.50, 0),
                    (46.50, 46.50, 0),
                    (46.50, -46.50, 0),
                    (-46.50, -46.50, 0)
                ],dtype="double")
                
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
                translacion = -np.matrix(rmat).T * np.matrix(translation_vector)
                
                print("\nRotation Yaw: ", -(angles[1]))
                print("Translation Z: ",translacion.item(2))
            
          
        cv2.imshow(window_name, im)
        
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.destroyWindow(window_name)