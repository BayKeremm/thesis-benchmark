#include "rektnet.hpp"

RektNet::RektNet(std::string filename) {
    try {
        rektnet = torch::jit::load(filename);
        std::cout << "success loading rektnet model\n";
}
    catch (const c10::Error& e) {
        std::cerr << "error loading rektnet model\n";
    }
}

// input frame is the cropped image
std::vector<std::pair<int,int>> RektNet::detect(const cv::Mat& coneFrame, cone_t& cone){

    // pre-process the image
    cv::Mat coneImage;
    cv::resize(coneFrame,coneImage,cv::Size(80,80));
    coneImage.convertTo(coneImage, CV_32FC3, 1.0 / 255.0);

    // convert to libtorch input tensor
    auto tensor = torch::from_blob(coneImage.data, { coneImage.rows, coneImage.cols, 3 }, torch::kFloat32);
    tensor = tensor.permute({ (2),(0),(1) });
    tensor.unsqueeze_(0); //add batch dim
    std::vector<torch::jit::IValue> input = std::vector<torch::jit::IValue>{tensor};
    
    // forward
    auto output = rektnet.forward(input).toTuple();
    auto points = output->elements()[1].toTensor();

    // extract coordinates
    std::vector<std::pair<int,int>> keypoints;
    for(int i=0; i < 7 ; i++){
        auto cord = points[0][i];
        double x = cord[0].item<double>();
        double y = cord[1].item<double>();
        keypoints.emplace_back(std::pair<int,int>(x * cone.box.width, y * cone.box.height));
    }

    // sort keypoints
    std::sort(keypoints.begin(), keypoints.end(), [](const std::pair<int,int> &a, const std::pair<int,int> &b) {return a.first < b.first;});

    return keypoints;
}