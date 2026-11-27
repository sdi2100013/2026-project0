#pragma once

#include <array>
#include <limits>
#include <map>
#include <string>
#include <vector>

#include "homography.hpp"
#include "rerank.hpp"

//αξιολόγηση των εκτιμομμενων ομογραφιών με τις ομογραφίες αναφοράς του HPatches.
//οι ομογραφίες αναφοράς χρησιμοποιούνται αποκληστικά για την αξιολόγηση.

//κατασκευή διαδρομής του αρχείου ομογραφίας αναφοράς
std::string groundTruthPath(const std::string& hpRoot, const std::string& dbId);

//ανάγνωση και έλεγχος εγκυρότητας της ομογραφίας αναφοράς
bool readGroundTruth(const std::string& path, Eigen::Matrix3d& H, std::string& error);


