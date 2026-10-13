#pragma once

#ifndef DATASET_HPP
#define DATASET_HPP
#include <string>
#include <opencv2/opencv.hpp>
#include <vector>


//Εξάγιε τους σιφτ descriptors από εικόνα
cv::Mat extractSIFT(const std::string& impath);

//Επιστρέφει το σύνολο των ακολουθιών που έχουν συγκεκιμένο setname πχ(validation, train, test)
std::vector<std::string> loadSplit(const std::string& spPath, const std::string& setName);

#endif