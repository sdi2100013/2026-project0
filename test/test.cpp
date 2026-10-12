#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
int main(int argc, char **argv)
{
	const char *path = argc > 1 ? argv[1] : "test/image.jpg";
	cv::Mat gray = cv::imread(path, cv::IMREAD_GRAYSCALE);
	if (gray.empty())
	{
		std::cerr << "Could not read image: " << path << '\n';
		return 1;
	}
	auto sift = cv::SIFT::create();
	std::vector<cv::KeyPoint> keypoints;
	cv::Mat descriptors;
	sift->detectAndCompute(gray, cv::noArray(), keypoints, descriptors);
	// One 128-dimensional descriptor per detected keypoint (CV_32F).
	std::cout << descriptors.rows << " x " << descriptors.cols << '\n';
	// std::cout << descriptors << std::endl;
	return 0;
}