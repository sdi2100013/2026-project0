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

//Επιστρέφει μια συγκεκιμένη διαδρομή προς συγκεκιρμένη εικόνα πχ. hppatches/i_autannes/1.ppm
std::string hpatchesImPath(const std::string& hpPath, const std::string& seqName, int imNumber);

//Επιστρέφει vector με τις 6 διαδορμές μιας ακολουθίας
std::vector<std::string> allhpatchesImPaths(const std::string& hpPath, const std::string& seqName);


#endif