#pragma once
#include <string>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include "utils.hpp"

class Handler3D{

    public:
        Handler3D() = default;
        Handler3D(std::string filename);
        void printParams();

        cv::Vec3s solvePnP(cone_t& cone);


    private:
        std::vector<cv::Point3f> smallConeModel;
        std::vector<cv::Point3f> largeConeModel;

        cv::Mat left_D;
        cv::Mat left_K;
        cv::Mat left_R;
        cv::Mat left_P;

        cv::Mat right_D;
        cv::Mat right_K;
        cv::Mat right_R;
        cv::Mat right_P;

        cv::Mat ROTATION; //right to left
        cv::Mat TRANSLATION;
};