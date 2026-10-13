#pragma once

#ifndef DATASET_HPP
#define DATASET_HPP
#include <string>
#include <opencv2/opencv.hpp>

//Εξάγιε τους σιφτ descriptors από εικόνα
cv::Mat extractSIFT(const std::string& impath);

#endif