#include <opencv2/highgui/highgui.hpp>

#include "yolo.hpp"
#include "rektnet.hpp"
#include "handler3d.hpp"
#include "utils.hpp"

// testing libraries
#include <chrono>
#include <string>
#include <fstream>

//static std::string yolo_path = "/home/rio/thesis_ws/src/inference_package/networks/yolo.torchscript";
//static Net_config YOLOV7_nets = { 0.8, 0.6, yolo_path, 640, 640};
//static YOLOV7 yolo(YOLOV7_nets);

//static std::string rektnet_path = "/home/rio/thesis_ws/src/inference_package/networks/rektnet.pt";
//static RektNet rektnet{rektnet_path};

//static std::string calibration_path{"/home/rio/thesis_ws/src/inference_package/camera_parameters/"};
//static Handler3D handler3d(calibration_path+"calibration.txt");

//static std::vector<std::string> class_names{"blue", "large orange", "small orange", "unkown", "yellow"};


int main(int argc, char ** argv){

    //read the video
    std::string fileName = "../../data/out.mp4";
    cv::VideoCapture cap(fileName);
    if(!cap.isOpened()){
        std::cout << "Error opening video stream or file" << std::endl;
        return -1;
    }
   
  while(1){
 
    cv::Mat frame;
    // Capture frame-by-frame
    cap >> frame;
  
    // If the frame is empty, break immediately
    if (frame.empty())
      break;
 
    // Display the resulting frame
    cv::imshow( "Frame", frame );
 
    // Press  ESC on keyboard to exit
    char c=(char)cv::waitKey(25);
    if(c==27)
      break;
  }
  
  // When everything done, release the video capture object
  cap.release();
 
  // Closes all the frames
  cv::destroyAllWindows();
   
  return 0;


}