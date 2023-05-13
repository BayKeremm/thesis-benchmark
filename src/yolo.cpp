#include "yolo.hpp"
#include <omp.h>

YOLOV7::YOLOV7(const Net_config config) {

	this->confThreshold = config.confThreshold;
	this->nmsThreshold = config.nmsThreshold;
    
    try {
        yolo = torch::jit::load(config.modelpath);
        std::cout << "success loading yolo model\n";
    }
    catch (const c10::Error& e) {
        std::cerr << "error loading yolo model\n";
    }
	
	this->inpHeight = config.inpHeight;
	this->inpWidth = config.inpWidth;
}

std::vector<cone_t> YOLOV7::detect(const cv::Mat& frame)
{
	// save size relations of initial frame
    float ratioh = (float) frame.rows / this->inpHeight;
    float ratiow = (float) frame.cols / this->inpWidth;

	// preprocess image
    cv::Mat image;
    cv::resize(frame,image,cv::Size(this->inpHeight,this->inpWidth)); // resize
    cv::cvtColor(image,image, cv::COLOR_RGB2BGR ); //opencv uses bgr
    image.convertTo(image, CV_32FC3, 1.0 / 255.0); // normalize

    // define as libtorch input tensor
    auto tensor = torch::from_blob(image.data, { image.rows, image.cols, 3 }, torch::kFloat32);
    tensor = tensor.permute({ (2),(0),(1) });
    tensor.unsqueeze_(0); // add batch dim (an inplace operation just like in pytorch)
    std::vector<torch::jit::IValue> input = std::vector<torch::jit::IValue>{tensor};
    
    // forward
    auto output = yolo.forward(input).toTuple();
    at::Tensor preds = output->elements()[0].toTensor();

    // filter boxes
	std::vector<float> confidences;
	std::vector<cv::Rect> boxes;
	std::vector<int> classIds;
    const int proposals = 25200; // default for 640x640 images

    omp_set_num_threads(4);
    #pragma omp parallel for
    for(int i = 1; i <= proposals; i++){
        at::Tensor pred = preds.slice(1,i-1,i);
        float box_score = pred.index({0,0,4}).item<float>();
        if(box_score > this->confThreshold){// larger than confidence threshold
            at::Tensor scores = pred.slice(2,5,10);
            float max = torch::max(scores).item<float>();
            at::Tensor max_index = (scores == max).nonzero();
            max *= box_score;
            
            if(max > this->confThreshold) { // conf threshold

                const int class_id = max_index.index({0,2}).item<int>();

                float cx = pred.index({0,0,0}).item<float>() * ratiow;
                float cy = pred.index({0,0,1}).item<float>() * ratioh;
                float w = pred.index({0,0,2}).item<float>() * ratiow;
                float h = pred.index({0,0,3}).item<float>() * ratioh;
                int left = int(cx - 0.5*w);
                int top = int(cy - 0.5*h);
                cv::Rect b = cv::Rect(left, top, (int)(w), (int)(h));

                #pragma omp critical
                {
                    confidences.push_back((float)max);
                    boxes.push_back(b);
                    classIds.push_back(class_id);
                }
            } 
        }
    }
	std::vector<int> indices;	
    std::vector<cone_t> cones;
	cv::dnn::NMSBoxes(boxes, confidences, this->confThreshold, this->nmsThreshold, indices);
    for (size_t i = 0; i < indices.size(); ++i)
	{
        // This is neccesarry because the detection goes out of the frame and you get seg fault
        int idx = indices[i];
        if(boxes[idx].x >= 0 && boxes[idx].y >= 0 &&
            boxes[idx].x + boxes[idx].width <= 1200 &&
            boxes[idx].y + boxes[idx].height <= 1000) {
            if(boxes[idx].width < boxes[idx].height) {
		        cones.push_back({boxes[idx],classIds[idx],confidences[idx]});
            }
        }
	}

    // filter 8 largest boxes not fallen over
    std::sort(cones.begin(), cones.end(), [](const cone_t& c1, const cone_t& c2) {
        return c1.box.height > c2.box.height;
    });

    std::vector<cone_t> filtered_cones; 
    int numBoxesAdded = 0;
    for (const auto& cone : cones) {
        if (cone.box.width <= cone.box.height && numBoxesAdded < 40) {
            filtered_cones.push_back(cone);
            numBoxesAdded++;
        }
    }

	return cones;
}