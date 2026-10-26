#pragma once

#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>

#include <random>
#include <string>
#include <vector>

//Διάσταση των SIFT descriptors
constexpr int kSiftDim = 128;

//Ρυθμίσεις SIFT κοινές για όλες τις εικόνες
//cv::SIFT::create(nfeatures, nOctaveLayers, contrastThreshold, edgeThreshold, sigma)
constexpr int kSiftFeatures = 0;
constexpr int kSiftOctaveLayers = 3;
constexpr double kSiftContrastThreshold = 0.01; 
constexpr double kSiftEdgeThreshold = 10; 
constexpr double kSiftSigma = 1.6;


//Δημιουργεί ένα μοναδικό αναγνωριστικό με βάση τις παραμέτρους του SIFT
std::string siftConfigId();

//Διαβάζει την εικόνα και εξάγει τους SIFT descriptors
bool extractSiftDescriptors(const std::string& path, cv::Mat& gray, std::vector<cv::KeyPoint>& keypoints, std::vector<float>& descriptors, std::string& error);

//Επιλέγει τυχαία μέχρι S descriptors από αυτούς που εξήχθησαν
std::vector<float> sampleDescriptors(const std::vector<float>& descriptors, int S, std::mt19937& rng);

struct TrainingStats {
    int sequences = 0;
    int images = 0;
    int emptyImages = 0;  // valid images with zero SIFT descriptors
    long before = 0;      // descriptors before sampling
    long after = 0;       // descriptors after sampling
};

//Επιστρέφει τα ονόματα των sequences του συνόλου set ("train", "validation" ή "test") με τη σειρά που εμφανίζονται στο αρχείο split.
//Επιστρέφει false και συμπληρώνει το error αν το αρχείο δεν μπορεί να διαβαστεί ή αν κάποια γραμμή του δεν είναι έγκυρη.
bool readSplitSequences(const std::string& splitPath, const std::string& set, std::vector<std::string>& names, std::string& error);

//Συγκεντρώνει τους SIFT descriptors από τις εικόνες εκπαίδευσης
bool collectTrainingDescriptors(const std::string& hpPath, const std::string& splitPath, int S, std::mt19937& rng, std::vector<float>& training, TrainingStats& stats, std::string& error);