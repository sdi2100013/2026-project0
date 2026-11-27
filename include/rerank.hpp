#pragma once

#include <limits>
#include <vector>

#include "feature_store.hpp"
#include "homography.hpp"
#include "matching.hpp"
#include "ransac.hpp"

//αποτελέσματα γεωμετρικής επαλήθευσης μιας υποψιφιας εικόνας
struct CandidateResult {
	int dbIndex = -1;    //δείκτης εικόνας στη βάση
	int matches = 0;     //πλήθος αντιστοιχήσεων μετά το Lowe ratio test
	bool valid = false;  //εγγυρότητα της τελικής ομογραφίας
	int inliers = 0;     //πλήθος τελικών inliers μετά την επανεκτίμηση
	double medianError = std::numeric_limits<double>::quiet_NaN();
	Eigen::Matrix3d H = Eigen::Matrix3d::Identity();
};

//γεωμετρική επαλήθευση υποψήφιας ηκόνας με Lowe ratio test και RANSAC.
//Ο δείκτης dbIndex συμπληρώνεται από τον καλούντα.
CandidateResult verifyCandidate(const ImageFeatures& query, const ImageFeatures& candidate,
                                const std::vector<TwoNearest>& nearest, double rho, const RansacParams& ransac);

//Επανακατάταξη με βάση το πλήθος των αποδεκτών matche
//σε ισοβαθμία διατηρείται η αρχική σειρά ανάκτησης
std::vector<int> rankByMatches(const std::vector<CandidateResult>& results);

//επανακατάταξη με βάση το πλήθος των τελικών inliers
//Σε ισοβαθμία προηγητε η μικρότερη διάμεσος σφάλματος επαναπροβολής.
//οι τιμές NaN τοποθετούνται τελευταίες στην ισοβαθμίσ
std::vector<int> rankByInliers(const std::vector<CandidateResult>& results);