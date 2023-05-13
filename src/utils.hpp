#pragma once
#include <torch/script.h>
#include <string>
#include <opencv2/opencv.hpp>

struct cone_t{
    cv::Rect box{};
    int classId{0};
    float confidence{0.0};
    std::vector<std::pair<int,int>> keypoints{};
    cv::Vec3s translation{0,0,0};
    bool valid = true;
};