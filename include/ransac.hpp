#pragma once

#include <limits>
#include <string>
#include <vector>

#include "homography.hpp"

// ransac για ομογραφια query → candidate, πανω στον normalized dlt
//
// σε καθε δοκιμη διαλεγουμε 4 διαφορετικα ζευγη και πεταμε τις κακες τετραδες.
// μετα υπολογιζουμε ομογραφια και μετραμε τους inliers (σφαλμα επαναπροβολης <
// tau) σε ολα τα M ζευγη. καλυτερη υποθεση ειναι αυτη με τους περισσοτερους
// inliers. αν ισοβαθμουν, κερδιζει η μικροτερη μεση τιμη (median) των σφαλματων
// των inliers. αν ειναι ενεργο το adaptive, το οριο δοκιμων μικραινει σε
// ceil(log(1-p) / log(1-w^4)), οπου w = (inliers της καλυτερης) / M.
// στο τελος η ομογραφια υπολογιζεται ξανα ΜΟΝΟ απο τους inliers της καλυτερης
// υποθεσης και οι τελικοι inliers ξαναβρισκονται σε ολα τα M ζευγη.

struct RansacParams {
	double tau = 3.0;  // οριο σφαλματος σε pixels (inlier αν σφαλμα < tau)
	int maxIterations =
	    2000;  // Tmax: μεγιστος αριθμος δοκιμων (μετραν και οι κακες τετραδες)
	double p = 0.99;       // πιθανοτητα επιτυχιας που θελουμε
	bool adaptive = true;  // αν ειναι true, το οριο δοκιμων μικραινει
	unsigned seed = 1;     // seed του std::mt19937 (δες ransacSeed)
};

struct RansacResult {
	bool valid = false;
	Eigen::Matrix3d H = Eigen::Matrix3d::Identity();  // τελικη H, μετα το refit
	std::vector<int> inliers;  // τελικοι inliers: δεικτες στα M ζευγη
	double medianError =
	    std::numeric_limits<double>::quiet_NaN();  // median των inliers, NaN αν
		                                           // δεν ειναι valid
};

// επιστρεφει true μονο αν result.valid. αλλιως το error λεει τον λογο
// (κακες παραμετροι, M < 4, μη σωστα δεδομενα, καμια σωστη υποθεση, αποτυχια
// του refit) και το result εχει αδεια λιστα inliers και medianError = NaN (η H
// δεν εχει νοημα)
bool ransacHomography(const std::vector<PointCorrespondence>& correspondences,
                      const RansacParams& params, RansacResult& result,
                      std::string& error);

// seed για καθε ζευγος (query, candidate) απο το βασικο -seed, με std::seed_seq
// (ορισμενο απο το προτυπο, ιδιο αποτελεσμα σε καθε υλοποιηση)
// ετσι το αποτελεσμα ενος ζευγους δεν εξαρτιεται απο τα αλλα ζευγη ουτε απο το
// C
unsigned ransacSeed(unsigned baseSeed, int queryId, int candidateId);