#include <opencv2/opencv.hpp>
#include "dataset.hpp"
#include <iostream>
#include <vector>
#include <stdexcept>
#include <fstream>


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
	//std::cout << descriptors.rows << " x " << descriptors.cols << '\n';
	// std::cout << descriptors << std::endl;
	return descriptors;
}


std::vector<std::string> loadSplit(const std::string& spPath, const std::string& setName){
    //αρχικοποίηση λίστας απο string
    std::vector<std::string> sequences;
    //ανοιγμα split
    std::ifstream file(spPath);
    
    if(!file.is_open()){
        throw std::runtime_error("Can't open split file: "+spPath);
    }

    std::string sequenceName;
    std::string spName;
    
    //διαβάζω ανα διάδα το περιεχόμενο του spli και αν έχουν ιδιο spName τα προσθέτω στη λίστα 
    while(file >> sequenceName >> spName){
        
        //*προσθήκη στο τέλος
        if(spName == setName){
            sequences.push_back(sequenceName);
        }
    }

    return sequences;
}

//Δεν ανοίγει εικόνα δημιουργεί μόνο το όνομα του αρχείου
std::string hpatchesImPath(const std::string& hpPath, const std::string& seqName, int imNumber){
    return hpPath + "/" + seqName + "/" + std::to_string(imNumber) + ".ppm";
}

std::vector<std::string> allhpatchesImPaths(const std::string& hpPath, const std::string& seqName){
    std::vector<std::string> paths;
    std::string temppath;
    
    for( int i = 1; i <= 6; i++){
        temppath = hpatchesImPath(hpPath, seqName, i);
        paths.push_back(temppath);
    }

    return paths;
}