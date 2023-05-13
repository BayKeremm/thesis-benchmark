#pragma once 
#include <torch/script.h>
#include <vector>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <fstream>
#include <sstream>
#include <iostream>
#include "utils.hpp"

struct Net_config
{
	float confThreshold; // Confidence threshold
	float nmsThreshold;  // Non-maximum suppression threshold
	std::string modelpath;
	int inpHeight;
	int inpWidth;
};

class YOLOV7
{
public:
	YOLOV7(const Net_config config);
	std::vector<cone_t> detect(const cv::Mat& frame);
	std::vector<std::string> class_names;
private:
    torch::jit::script::Module yolo;
	int inpWidth;
	int inpHeight;
	int num_class;

	float confThreshold;
	float nmsThreshold;
};