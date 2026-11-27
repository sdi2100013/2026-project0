#include "hpatches_gt.hpp"

#include <cmath>
#include <fstream>

// Κατασκευή διαδρομής του αρχείου ομογραφίας αναφοράς
std::string groundTruthPath(const std::string& hpRoot, const std::string& dbId) {
	const size_t slash = dbId.find('/');
	//ελεγχος της βασικής μορφής του αναγνοριστικού εικόνας
	if (slash == std::string::npos || slash == 0 || slash == dbId.size() - 1) {
		return "";
	}
	const std::string sequence = dbId.substr(0, slash);
	const std::string image = dbId.substr(slash + 1);
	return hpRoot + "/" + sequence + "/H_1_" + image;
}



//ανάγνωση της ομογραφίας αναφοράς από το αντίστοιχο αρχείο
bool readGroundTruth(const std::string& path, Eigen::Matrix3d& H, std::string& error) {
	std::ifstream in(path);
	if (!in) {
		error = "cannot open ground-truth homography '" + path + "'";
		return false;
	}
	Eigen::Matrix3d matrix;
	//ανάγνοση των 9 στοιχείων του πίνακα ομογραφίας
	for (int i = 0; i < 3; ++i) {
		for (int j = 0; j < 3; ++j) {
			if (!(in >> matrix(i, j)) || !std::isfinite(matrix(i, j))) {
				error = "'" + path + "' does not contain 9 finite numbers";
				return false;
			}
		}
	}
	//ελεγχος για επιπλέον περιεχόμενο μετά τα 9 στοιχεία
	std::string extra;
	if (in >> extra) {
		error = "'" + path + "' contains extra content";
		return false;
	}
	//ελεγχος εγγηρότητας της ομογραφίας
	if (!isValidHomography(matrix)) {
		error = "ground-truth homography '" + path + "' is invalid";
		return false;
	}
	H = matrix;
	return true;
}


