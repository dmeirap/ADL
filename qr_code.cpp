#include <iostream>
#include <zbar.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/calib3d/calib3d.hpp"
#include <opencv2/imgproc/imgproc.hpp>
#include <iomanip>
#include <ctime>  
#include <time.h>
#include "matplotlibcpp.h"

namespace plt = matplotlibcpp;
using namespace std;
using namespace cv;
using namespace zbar;


int main(int argc, const char *argv[])
{
	VideoCapture cap("../videos/Video_Prueba_Dron_1200_config1.mp4");
	//VideoCapture cap(2);
	cap.set(cv::CAP_PROP_FRAME_WIDTH, 1920);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 1080);
	//cap.set(cv::CAP_PROP_BRIGHTNESS, -64);
	//cap.set(cv::CAP_PROP_CONTRAST, 5);
	//cap.set(cv::CAP_PROP_SATURATION, 0);
	//cap.set(cv::CAP_PROP_GAMMA, 80);
	//cap.set(cv::CAP_PROP_SHARPNESS, 0);
	
	time_t start, end;
    int counter = 0;
    double sec;
    double fps;
	

    if (!cap.isOpened()) { // check if we succeeded
      cout << "Failed to open the camera" << endl;
      return -1;
    }
	
	namedWindow("Deteccion - QR", WINDOW_AUTOSIZE);

	ImageScanner scanner;
	scanner.set_config(ZBAR_NONE, ZBAR_CFG_ENABLE, 1);

	//float yaw_values = collections.deque(maxlen=50);

	while (true) {
        
        Mat frame, frame_grayscale;
        cap.read(frame);


		if (counter == 0){
            time(&start);
        }

		time(&end);
        counter++;
        sec = difftime(end, start);
        fps = counter/sec;
        if (counter > 30)
            putText(frame, "FPS: " + to_string(int(fps)), { 50, 50 }, FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255),2);
        if (counter == (INT_MAX - 1000))
            counter = 0;

		

        cvtColor(frame, frame_grayscale, COLOR_BGR2GRAY);

        // Obtain image data
        int width = frame.cols;
        int height = frame.rows;
        uchar *raw = (uchar *)(frame_grayscale.data);

        // Wrap image data
        Image image(width, height, "Y800", raw, width * height);

        scanner.scan(image);

		vector<Point3f> model_points;
		//model_points.push_back(Point3f(0.0, 0.0, 0.0));
		model_points.push_back(Point3f(-46.50, 46.50, 0));
		model_points.push_back(Point3f(46.50, 46.50, 0));
		model_points.push_back(Point3f(46.50, -46.50, 0)); 
		model_points.push_back(Point3f(-46.50, -46.50, 0));
		//model_points.push_back(Point3f(0, 46.50, 0));  
		//model_points.push_back(Point3f(46.50, 0, 0)); 
		//model_points.push_back(Point3f(0, -46.50, 0));

		Mat camera_matrix(3,3,cv::DataType<double>::type);
		camera_matrix.at<double>(0,0) = 1.51187075e+03;
		camera_matrix.at<double>(0,1) = 0;
		camera_matrix.at<double>(0,2) = 9.73619124e+02;
		camera_matrix.at<double>(1,0) = 0;
		camera_matrix.at<double>(1,1) = 1.51847952e+03;
		camera_matrix.at<double>(1,2) = 3.55830121e+02;
		camera_matrix.at<double>(2,0) = 0;
		camera_matrix.at<double>(2,1) = 0;
		camera_matrix.at<double>(2,2) = 1;

		Mat dist_coeffs(5,1,cv::DataType<double>::type);
		dist_coeffs.at<double>(0) = -2.33996748e-02;
		dist_coeffs.at<double>(1) = 7.68975435e-01;
		dist_coeffs.at<double>(2) = -9.51846221e-03;
		dist_coeffs.at<double>(3) = 1.27593135e-03;
		dist_coeffs.at<double>(4) = -1.79462157e+00;							  

        for (Image::SymbolIterator symbol = image.symbol_begin(); symbol != image.symbol_end(); ++symbol) {
            

			Point2f(bottomLeftCorners) = Point2f(symbol->get_location_x(0), symbol->get_location_y(0));
			Point2f(topLeftCorners) = Point2f(symbol->get_location_x(1), symbol->get_location_y(1));
            Point2f(topRightCorners) = Point2f(symbol->get_location_x(2), symbol->get_location_y(2));
			Point2f(bottomRightCorners) = Point2f(symbol->get_location_x(3), symbol->get_location_y(3));
			
            // Draw location of the symbols found
            if (symbol->get_location_size() == 4) {
                //rectangle(frame, Rect(symbol->get_location_x(i), symbol->get_location_y(i), 10, 10), Scalar(0, 255, 0));
                line(frame, Point2f(topRightCorners), Point2f(bottomRightCorners), Scalar(0, 0, 255), 2, 8, 0);
                line(frame, Point2f(bottomRightCorners), Point2f(bottomLeftCorners), Scalar(0, 0, 255), 2, 8, 0);
                line(frame, Point2f(bottomLeftCorners), Point2f(topLeftCorners), Scalar(0, 0, 255), 2, 8, 0);
                line(frame, Point2f(topLeftCorners), Point2f(topRightCorners), Scalar(0, 0, 255), 2, 8, 0);
            }

			vector<Point2f> image_points;
			//image_points.push_back(Point2f((symbol->get_location_x(3) + symbol->get_location_x(2))/2,symbol->get_location_y(3) + symbol->get_location_y(2))/2); //(0.0, 0.0, 0.0)
			image_points.push_back(Point2f(topLeftCorners)); //(-46.50, 46.50, 0)
			image_points.push_back(Point2f(topRightCorners)); //(46.50, 46.50, 0)
			image_points.push_back(Point2f(bottomRightCorners)); //(46.50, -46.50, 0)
			image_points.push_back(Point2f(bottomLeftCorners)); //(-46.50, -46.50, 0)
			//image_points.push_back(Point2f((symbol->get_location_x(3) + symbol->get_location_x(0))/2,symbol->get_location_y(3) + symbol->get_location_y(0))/2); //(0, 46.50, 0)
			//image_points.push_back(Point2f((symbol->get_location_x(0) + symbol->get_location_x(1))/2,symbol->get_location_y(0) + symbol->get_location_y(1))/2); //(46.50, 0, 0)
			//image_points.push_back(Point2f((symbol->get_location_x(2) + symbol->get_location_x(1))/2,symbol->get_location_y(2) + symbol->get_location_y(1))/2); //(0, -46.50, 0)

			Mat rvec(3,1,cv::DataType<double>::type);
  			Mat tvec(3,1,cv::DataType<double>::type);

			cv::TermCriteria criteria = cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 10, 0.001);
			cv::Size winSize =  cv::Size( 11, 11 );
    		cv::Size zeroZone = cv::Size( -1, -1 );
         	cv::cornerSubPix(frame_grayscale,image_points,winSize,zeroZone,criteria);

			solvePnP(model_points, image_points, camera_matrix, dist_coeffs, rvec, tvec, false, SOLVEPNP_IPPE_SQUARE);

			//*********EJES DE COORDENADAS*************

			/*Mat unitv_points(4,3,cv::DataType<double>::type);
			unitv_points.at<double>(0,0) = 0;
			unitv_points.at<double>(0,1) = 0;
			unitv_points.at<double>(0,2) = 0;
			unitv_points.at<double>(1,0) = 46.50;
			unitv_points.at<double>(1,1) = 0;
			unitv_points.at<double>(1,2) = 0;
			unitv_points.at<double>(2,0) = 0;
			unitv_points.at<double>(2,1) = 46.50;
			unitv_points.at<double>(2,2) = 0;
			unitv_points.at<double>(3,0) = 0;
			unitv_points.at<double>(3,1) = 0;
			unitv_points.at<double>(3,2) = 46.50;
			Mat axis_points;
			projectPoints(unitv_points, rvec, tvec, camera_matrix, dist_coeffs, axis_points);
			vector<cv::Scalar> colors = { Scalar(255, 0, 0), Scalar(0, 255, 0), Scalar(0, 0, 255), Scalar(0, 0, 0) };
			if (axis_points.rows > 0) {
				axis_points = axis_points.reshape(4, 2);
				cv::Point origin(axis_points.at<double>(0, 0), axis_points.at<double>(0, 1));
				for (int i = 1; i < 4; i++) {
					cv::Point p(axis_points.at<double>(i, 0), axis_points.at<double>(i, 1));
					if (origin.x > 5 * frame.cols || origin.y > 5 * frame.cols) break;
					if (p.x > 5 * frame.cols || p.y > 5 * frame.cols) break;
					cv::line(frame, origin, p, colors[i-1], 5);
				}
			}*/
			
			Mat rmat(3,3,cv::DataType<double>::type);
			Mat transpose(3,3,cv::DataType<double>::type);
			Vec3d angles;
			Mat mtxR(3,3,cv::DataType<double>::type);
			Mat mtxQ(3,3,cv::DataType<double>::type);
			Mat translacion(1,3,cv::DataType<double>::type);

			Rodrigues(rvec,rmat);

			for (int i = 0; i < 3; ++i)
      			for (int j = 0; j < 3; ++j) {
         			transpose.at<double>(j,i) = rmat.at<double>(i,j);
      			}

			angles = RQDecomp3x3(transpose, mtxR, mtxQ);
			translacion = transpose * tvec;


			if (15 < angles(0) && angles(0) < 25 ) {
				
				cout    << "\n\nDecoded: " << symbol->get_type_name()
						<< " Symbol: \"" << symbol->get_data() << '"' << endl;
				cout << "\nRotacion Yaw: " << angles(1) << endl;
				cout << "Translacion Z: " << translacion.at<double>(2,0) << endl;
				
        	}
		}

		Size reSize =  Size( 640, 400 );
		resize(frame, frame, reSize);
		// Show captured frame, now with overlays!
        imshow("Deteccion - QR", frame);
        
         if (waitKey(30) == 'q')  
		{    
			break;   
		}  
    }














































return 0;

}
