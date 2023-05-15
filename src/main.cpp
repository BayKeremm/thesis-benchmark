#include <opencv2/highgui/highgui.hpp>

#include "yolo.hpp"
#include "rektnet.hpp"
#include "handler3d.hpp"
#include "utils.hpp"

// testing libraries
#include <chrono>
#include <string>
#include <fstream>

static std::string yolo_path = "../../networks/yolov7.torchscript.pt";
static Net_config YOLOV7_nets = { 0.8, 0.6, yolo_path, 640, 640};
static YOLOV7 yolo(YOLOV7_nets);

static std::string rektnet_path = "../../networks/rektnet.pt";
static RektNet rektnet{rektnet_path};

static std::string calibration_path{"../../camera/calibration.txt"};
static Handler3D handler3d(calibration_path);

static std::vector<std::string> class_names{"blue", "large orange", "small orange", "unkown", "yellow"};

void drawPred(const cone_t& cone, cv::Mat& frame)   
{
    int left = cone.box.x;
    int right = cone.box.width + cone.box.x;
    int top = cone.box.y;
    int bottom = cone.box.height + cone.box.y;

    // color based on classId or not valid
    cv::Scalar color;
    if(!cone.valid == true) {
        color = cv::Scalar(0, 0, 255); // Red is (0, 0, 255) in OpenCV
    } 
    else if (cone.classId == 0) { // blue
        color = cv::Scalar(255, 0, 0);
    } else if (cone.classId == 4) { // yellow
        color = cv::Scalar(0, 255, 255);
    } else if (cone.classId == 1) { // large orange
        color = cv::Scalar(0, 100, 255);
    } else if (cone.classId == 2) { // small orange
        color = cv::Scalar(0, 200, 255);
    } else { // unkown set black, shouldnt happen!
        color = cv::Scalar(0, 0, 0);
    }

	// draw bounding box
	rectangle(frame, cv::Point(left, top), cv::Point(right, bottom), color, 2);

	// display class and confidence on top
    std::string class_confidence_text = class_names[cone.classId] + " c:" + cv::format("%.2f", cone.confidence);
    putText(frame, class_confidence_text, cv::Point(left, top-10), cv::FONT_HERSHEY_SIMPLEX, 0.30, color, 1);

	// display translation vector at bottom
    std::string translation_text = cv::format("x: %d, y: %d, z: %d ", cone.translation[0], cone.translation[1], cone.translation[2]);
    putText(frame, translation_text, cv::Point(left, bottom+10), cv::FONT_HERSHEY_SIMPLEX, 0.30, color, 1);
    
    // draw keypoints
    for(auto p: cone.keypoints){
		cv::circle(frame,cv::Point2d(p.first+left, p.second+top),2, color, -1);
	}
}


int main(int argc, char ** argv){
    cv::Mat frame = cv::imread("frame.jpeg");
    cv::cvtColor(frame,frame, cv::COLOR_RGB2BGR ); //opencv uses bgr

    auto t1 = std::chrono::high_resolution_clock::now();

    std::vector<cone_t> cones = yolo.detect(frame);

    auto t2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> ms_double = t2 - t1;
    std::cout << ms_double.count() << "yolo in total took \n";

    //t1 = std::chrono::high_resolution_clock::now();
    for(cone_t & cone: cones){
      cv::Mat coneImage = frame(cone.box);
      cone.keypoints = rektnet.detect(coneImage, cone);
      cone.translation = handler3d.solvePnP(cone);
    }
    //t2 = std::chrono::high_resolution_clock::now();
    //ms_double = t2 - t1;
    //std::cout << ms_double.count() << "rektnet took \n";
    for(cone_t & cone: cones){
      drawPred(cone,frame);
    }
    cv::imwrite( "out.jpeg", frame );

    ////read the video
    //std::string inputFile = "../../data/short.mp4";
    //cv::VideoCapture video(inputFile);
    ////std::string outputFile = "../../data/detected.mp4";
    ////cv::VideoWriter outputVideo(outputFile, cv::VideoWriter::fourcc('H','2','6','4'), 
        ////video.get(cv::CAP_PROP_FPS), cv::Size(video.get(cv::CAP_PROP_FRAME_WIDTH), video.get(cv::CAP_PROP_FRAME_HEIGHT)));

    //if(!video.isOpened()){
        //std::cout << "Error opening video stream or file" << std::endl;
        //return -1;
    //}
   
  //while(1){
 
    //cv::Mat frame;
    //// Capture frame-by-frame
    //video >> frame;
  
    //// If the frame is empty, break immediately
    //if (frame.empty())
      //break;

    //std::vector<cone_t> cones = yolo.detect(frame);

    //for(cone_t & cone: cones){
      //cv::Mat coneImage = frame(cone.box);
      //cone.keypoints = rektnet.detect(coneImage, cone);
      //cone.translation = handler3d.solvePnP(cone);
    //}

    ////for(cone_t & cone: cones){
      ////drawPred(cone,frame);
    ////}

    //// Display the resulting frame
    ////cv::imshow( "Frame", frame );
    ////outputVideo.write(frame);
 
    //// Press  ESC on keyboard to exit
    ////char c=(char)cv::waitKey(25);
    ////if(c==27)
      ////break;
  //}
  
  //// When everything done, release the video capture object
  //video.release();
  ////outputVideo.release();
 
  //// Closes all the frames
  cv::destroyAllWindows();
   
  return 0;


}