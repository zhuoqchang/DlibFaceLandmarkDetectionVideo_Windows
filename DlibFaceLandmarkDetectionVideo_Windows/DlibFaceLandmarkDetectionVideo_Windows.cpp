
	// The contents of this file are in the public domain. See LICENSE_FOR_EXAMPLE_PROGRAMS.txt
	/*
	This example program shows how to find frontal human faces in an image and
	estimate their pose.  The pose takes the form of 68 landmarks.  These are
	points on the face such as the corners of the mouth, along the eyebrows, on
	the eyes, and so forth.
	This example is essentially just a version of the face_landmark_detection_ex.cpp
	example modified to use OpenCV's VideoCapture object to read from a camera instead
	of files.
	Finally, note that the face detector is fastest when compiled with at least
	SSE2 instructions enabled.  So if you are using a PC with an Intel or AMD
	chip then you should enable at least SSE2 instructions.  If you are using
	cmake to compile this program you can enable them by using one of the
	following commands when you create the build project:
	cmake path_to_dlib_root/examples -DUSE_SSE2_INSTRUCTIONS=ON
	cmake path_to_dlib_root/examples -DUSE_SSE4_INSTRUCTIONS=ON
	cmake path_to_dlib_root/examples -DUSE_AVX_INSTRUCTIONS=ON
	This will set the appropriate compiler options for GCC, clang, Visual
	Studio, or the Intel compiler.  If you are using another compiler then you
	need to consult your compiler's manual to determine how to enable these
	instructions.  Note that AVX is the fastest but requires a CPU from at least
	2011.  SSE4 is the next fastest and is supported by most current machines.
	*/

#include <dlib/opencv.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
#include <dlib/gui_widgets.h>

#include <iostream>
#include <opencv2\opencv.hpp>

using namespace dlib;
using namespace std;

void rot90(cv::Mat &matImage, int rotflag) {
	//1=CW, 2=CCW, 3=180
	if (rotflag == 1) {
		transpose(matImage, matImage);
		flip(matImage, matImage, 1); //transpose+flip(1)=CW
	}
	else if (rotflag == 2) {
		flip(matImage, matImage, -1);    //flip(-1)=180          
	}
	else if (rotflag == 3) {
		transpose(matImage, matImage);
		flip(matImage, matImage, 0); //transpose+flip(0)=CCW     
	}
	else if (rotflag != 0) { //if not 0,1,2,3:
		cout << "Unknown rotation flag(" << rotflag << ")" << endl;
	}
}

int main(int argc, char** argv) {

	// This example takes in a shape model file and then a list of images to
	// process.  We will take these filenames in as command line arguments.
	// Dlib comes with example images in the examples/faces folder so give
	// those as arguments to this program.

	// Video orientation
	int rotflag;
	bool fullscreen = false;
	if (argc == 3) {
		rotflag = 0;
	}
	else if (argc == 4) {
		rotflag = atoi(argv[3]);
	}
	else if (argc == 5) {
		rotflag = atoi(argv[3]);
		fullscreen = true;
	}
	else {
		cout << "Input error!" << endl;
		return 0;
	}

	cv::VideoCapture cap(argv[1]);
	if (!cap.isOpened()) {
		cerr << "Unable to open " << argv[1] << endl;
		return 1;
	}

	// Open file to write landmarks
	ofstream outfile;
	std::string outputPath = std::string(argv[2]) + "\\landmarks.txt";
	outfile.open(outputPath);

	cv::Mat im, im_small;
	float downsampleRatio = 0.5;

	// Load face detection and pose estimation models.
	frontal_face_detector detector = get_frontal_face_detector();
	shape_predictor sp;
	deserialize("../../data/shape_predictor_68_face_landmarks.dat") >> sp;

	int count = 0;
	std::vector<rectangle> faces;

	std::string window_name = "DLib Face Detector";
	cv::namedWindow(window_name, cv::WINDOW_NORMAL);

	// Grab and process frames until the main window is closed by the user.
	while (true) {
		// Grab a frame
		if (!cap.read(im)) {
			std::cout << "Unable to retrieve frame from video stream." << std::endl;
			break;
		}
		// Rotate image
		rot90(im, rotflag);
		// Resize image for face detection
		cv::resize(im, im_small, cv::Size(), downsampleRatio, downsampleRatio);

		// Change to dlib's image format. No memory is copied.
		cv_image<bgr_pixel> cimg_small(im_small);
		cv_image<bgr_pixel> cimg(im);

		// Now tell the face detector to give us a list of bounding boxes
		// around all the faces in the image.
		if (fullscreen) {
			cout << "Frame " << count << endl;
			outfile << "Frame " << count;
			rectangle r(
				(long)(0),
				(long)(0),
				(long)(im.cols),
				(long)(im.rows)
			);
			// Landmark detection on full sized image
			full_object_detection shape = sp(cimg, r);
			for (int j = 0; j < shape.num_parts(); j++) {
				outfile << " " << shape.part(j);
			}
		}
		else {
			std::vector<rectangle> faces = detector(cimg_small);
			cout << "Frame " << count << ": " << faces.size() << " face(s) detected" << endl;

			// Now we will go ask the shape_predictor to tell us the pose of
			// each face we detected.
			outfile << "Frame " << count;
			if (faces.size() > 0) {
				// Resize obtained rectangle for full resolution image. 
				rectangle r(
					(long)(faces[0].left() / downsampleRatio),
					(long)(faces[0].top() / downsampleRatio),
					(long)(faces[0].right() / downsampleRatio),
					(long)(faces[0].bottom() / downsampleRatio)
				);
				// Landmark detection on full sized image
				full_object_detection shape = sp(cimg, r);
				for (int j = 0; j < shape.num_parts(); j++) {
					outfile << " " << shape.part(j);
				}
			}
		}
		cv::imshow(window_name, im);
		cv::waitKey(5);

		outfile << endl;
		count++;
	}
	outfile.close();
}