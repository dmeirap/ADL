import cv2
import numpy as np 

qr = cv2.QRCodeDetector()
cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
window_name = 'OpenCV QR Code'
cv2.namedWindow(window_name, 0)
#fourcc = cv2.VideoWriter_fourcc(*'MP4V')
out = cv2.VideoWriter('output.mp4', 0x00000021, 5.0, (1920, 1080))

while True:
    ret, im = cap.read()
    out.write(im)
     
    if ret:
        ret_qr, decoded_info, qr_points, _ = qr.detectAndDecodeMulti(im)
        if ret_qr:
            for s, p in zip(decoded_info, qr_points):
                
                if s:
                    print("\nNumero QR: ", s)
                    color = (0, 255, 0)
                else:
                    color = (0, 0, 255)
                
                im = cv2.polylines(im, [p.astype(int)], True, color, 6)
            
                topLeftCorners = qr_points[0][3]
                topRightCorners = qr_points[0][2]
                bottomLeftCorners = qr_points[0][0]    
                bottomRightCorners  = qr_points[0][1]
                
                image_points = np.array([
                    (int((topLeftCorners[0]+topRightCorners[0])/2),
                    int((topLeftCorners[1]+bottomLeftCorners[1])/2)),     # Nose
                    topLeftCorners,   
                    topRightCorners,     
                    bottomLeftCorners,     
                    bottomRightCorners      
                ], dtype="double") 
                
                model_points = np.array([
                    (0.0, 0.0, 0.0), 
                    # Left top corner
                    (-46.50, 46.50, 0),
                    # Right top corner
                    (46.50, 46.50, 0),
                    # Left bottom corner
                    (-46.50, -46.50, 0),
                    # Right bottom corner
                    (46.50, -46.50, 0)
                ],dtype="double")
                
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
                
                #print("\nRotation Roll: {0}".format(angles[0]))
                print("\nRotation Yaw: {0}".format(angles[1]))
                #print("\nRotation Pitch: {0}".format(angles[2]))
                #print("\nTranslation X: {0}".format(translation_vector[0]))
                #print("\nTranslation Y: {0}".format(translation_vector[1]))
                #print("\nTranslation Z: {0}".format(translation_vector[2]))
                print("\nTranslation:{0}".format(translacion[2]))
                
                
        cv2.imshow(window_name, im)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
out.release()
cv2.destroyWindow(window_name)