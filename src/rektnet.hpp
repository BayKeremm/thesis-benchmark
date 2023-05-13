#pragma once
#include <torch/script.h>
#include <opencv2/opencv.hpp>
#include "utils.hpp"

class RektNet{
    public:
        RektNet(std::string filename);
        std::vector<std::pair<int,int>> detect(const cv::Mat& frame, cone_t& cone);

    private:
        torch::jit::script::Module rektnet;
};