#include "handler3d.hpp"
#include <fstream>

cv::Mat parse_line(std::string line ){
    std::string temp = line.substr(line.find("[") + 1, line.find("]") - line.find("[") - 1);
    std::stringstream ss(temp);
    double value;
    std::vector<double> values;
    while (ss >> value)
    {
        values.push_back(value);
        if (ss.peek() == ',') ss.ignore();
    }
    // Create a cv::Mat object from the vector of values
    cv::Mat mat(1,values.size(), CV_64FC1);
    for(int i=0; i < values.size();i++){
        mat.at<double>(0,i) = values[i];
    }
    return mat;
}

void Handler3D::printParams(){
    std::cout << "D_left: " << left_D << std::endl;
    std::cout << "K_left: " << left_K << std::endl;
    std::cout << "R_left: " << left_R << std::endl;
    std::cout << "P_left: " << left_P << std::endl;
    std::cout << "D_right: " << right_D << std::endl;
    std::cout << "K_right: " << right_K << std::endl;
    std::cout << "R_right: " << right_R << std::endl;
    std::cout << "P_right: " << right_P << std::endl;
    std::cout << "TRANSLATION: " << TRANSLATION << std::endl;
    std::cout << "ROTATION: " << ROTATION << std::endl;
}

Handler3D::Handler3D(std::string filename){

    try{
        std::fstream data;
        data.open(filename);
        std::string l;
        if(data.is_open()){
            while(getline(data, l)){
                if(l == "Left:"){
                    getline(data, l);
                    left_D = parse_line(l);
                    getline(data, l);
                    left_K = parse_line(l);
                    getline(data, l);
                    left_R = parse_line(l);
                    getline(data, l);
                    left_P = parse_line(l);
                }else if(l == "Right:"){
                    getline(data, l);
                    right_D = parse_line(l);
                    getline(data, l);
                    right_K = parse_line(l);
                    getline(data, l);
                    right_R = parse_line(l);
                    getline(data, l);
                    right_P = parse_line(l);
                    getline(data, l);
                    TRANSLATION = parse_line(l);
                    getline(data, l);
                    ROTATION = parse_line(l);
                }else{
                    //std::cout << "other line" << "\n";
                }
            } 
            data.close();
        }
    }catch (cv::Exception &e){
        std::cout << "ERROR in reading calibration file" << std::endl; 
	}
    
    // define 3d points wrt to the base of the cone which is taken as the world frame
    // assume cones have jeypoints at 1/3 of their height as estimated by rektnet
    // small cone
    int cone_height = 32;
    int cone_radius = 11;
    smallConeModel.push_back(cv::Point3i(0,-cone_radius,0));
    smallConeModel.push_back(cv::Point3i(0,round(-cone_radius*2/3),round(cone_height*1/3)));
    smallConeModel.push_back(cv::Point3i(0,round(-cone_radius*1/3),round(cone_height*2/3)));
    smallConeModel.push_back(cv::Point3i(0,0,cone_height));
    smallConeModel.push_back(cv::Point3i(0,round(cone_radius*1/3),round(cone_height*2/3)));
    smallConeModel.push_back(cv::Point3i(0,round(cone_radius*2/3),round(cone_height*1/3)));
    smallConeModel.push_back(cv::Point3i(0,cone_radius,0));
    // large cone
    cone_height = 50;
    cone_radius = 14;
    largeConeModel.push_back(cv::Point3i(0,-cone_radius,0));
    largeConeModel.push_back(cv::Point3i(0,round(-cone_radius*2/3),round(cone_height*1/3)));
    largeConeModel.push_back(cv::Point3i(0,round(-cone_radius*1/3),round(cone_height*2/3)));
    largeConeModel.push_back(cv::Point3i(0,0,cone_height));
    largeConeModel.push_back(cv::Point3i(0,round(cone_radius*1/3),round(cone_height*2/3)));
    largeConeModel.push_back(cv::Point3i(0,round(cone_radius*2/3),round(cone_height*1/3)));
    largeConeModel.push_back(cv::Point3i(0,cone_radius,0));
}

cv::Vec3s Handler3D::solvePnP(cone_t& cone)
{
    //corresponding 2d points
    std::vector<cv::Point2f> imagePoints;
    for(std::pair<int,int> & p : cone.keypoints){
        imagePoints.push_back(cv::Point2f(p.first+cone.box.x,p.second+cone.box.y));
    }

    cv::Mat cam_mat = this->left_K.reshape(0,3); 

    cv::Mat rvec, tvec;
    // check if its a large cone
    if(cone.classId == 1) {
        cv::solvePnP(largeConeModel, imagePoints, cam_mat, this->left_D, rvec, tvec);
    } 
    else cv::solvePnP(smallConeModel, imagePoints, cam_mat, this->left_D, rvec, tvec);

    short int x = round(tvec.at<double>(0,0)); 
    short int y = round(tvec.at<double>(1,0)); 
    short int z = round(tvec.at<double>(2,0));

    if(x < -500 || x > 500) cone.valid = false; // if cone is horizontally bellow or above threshold (left/right)
    else if(y < -150 || y > 150) cone.valid = false; // if cone is vertically bellow or above threshold (down/up)
    else if(z < 50 || z > 2000) cone.valid = false; // if cone is less than or beyond reasonable distance

    cv::Vec3s translation{x, y, z};

    return translation;
}
