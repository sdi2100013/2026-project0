#include <opencv2/opencv.hpp>
#include "dataset.hpp"
#include <iostream>
#include <vector>
#include <stdexcept>


//Συνάρτηση sift από διαφάνειες
cv::Mat extractSIFT(const std::string& impath){
	cv::Mat gray = cv::imread(impath, cv::IMREAD_GRAYSCALE);
	if (gray.empty()){
		throw std::runtime_error("Couldn't read image: "+impath);
	}
	auto sift = cv::SIFT::create();
	std::vector<cv::KeyPoint> keypoints;
	cv::Mat descriptors;
	sift->detectAndCompute(gray, cv::noArray(), keypoints, descriptors);
	// One 128-dimensional descriptor per detected keypoint (CV_32F).
	std::cout << descriptors.rows << " x " << descriptors.cols << '\n';
	// std::cout << descriptors << std::endl;
	return descriptors;
}

