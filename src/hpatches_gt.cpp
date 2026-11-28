#include "hpatches_gt.hpp"

#include <cmath>
#include <fstream>

// Κατασκευή διαδρομής του αρχείου ομογραφίας αναφοράς
std::string groundTruthPath(const std::string& hpRoot,
                            const std::string& dbId) {
	const size_t slash = dbId.find('/');
	// ελεγχος της βασικής μορφής του αναγνοριστικού εικόνας
	if (slash == std::string::npos || slash == 0 || slash == dbId.size() - 1) {
		return "";
	}
	const std::string sequence = dbId.substr(0, slash);
	const std::string image = dbId.substr(slash + 1);
	return hpRoot + "/" + sequence + "/H_1_" + image;
}

// ανάγνωση της ομογραφίας αναφοράς από το αντίστοιχο αρχείο
bool readGroundTruth(const std::string& path, Eigen::Matrix3d& H,
                     std::string& error) {
	std::ifstream in(path);
	if (!in) {
		error = "cannot open ground-truth homography '" + path + "'";
		return false;
	}
	Eigen::Matrix3d matrix;
	// ανάγνοση των 9 στοιχείων του πίνακα ομογραφίας
	for (int i = 0; i < 3; ++i) {
		for (int j = 0; j < 3; ++j) {
			if (!(in >> matrix(i, j)) || !std::isfinite(matrix(i, j))) {
				error = "'" + path + "' does not contain 9 finite numbers";
				return false;
			}
		}
	}
	// ελεγχος για επιπλέον περιεχόμενο μετά τα 9 στοιχεία
	std::string extra;
	if (in >> extra) {
		error = "'" + path + "' contains extra content";
		return false;
	}
	// ελεγχος εγγηρότητας της ομογραφίας
	if (!isValidHomography(matrix)) {
		error = "ground-truth homography '" + path + "' is invalid";
		return false;
	}
	H = matrix;
	return true;
}

// κατασκευή των τεσάρων γονιών της αρχικής εικόνας
std::array<Point2, 4> imageCorners(double width, double height) {
	return {{{0, 0}, {width, 0}, {width, height}, {0, height}}};
}

namespace {
// υπολογισμός του μέσου σφάλματος προβολής των τεσσάρων γωνιών
bool cornerError(const Eigen::Matrix3d& estimated,
                 const Eigen::Matrix3d& groundTruth,
                 const std::array<Point2, 4>& corners, double& error) {
	double sum = 0;
	for (const Point2& corner : corners) {
		Point2 estimatedPoint;
		Point2 groundTruthPoint;
		// προβολή της ίδιας γονίας με τις δύο ομογραφίες
		if (!projectPoint(estimated, corner, estimatedPoint) ||
		    !projectPoint(groundTruth, corner, groundTruthPoint)) {
			return false;
		}
		// ευκλείδια απόσταση μεταξύ των δύο προβολών
		sum += std::hypot(estimatedPoint.x - groundTruthPoint.x,
		                  estimatedPoint.y - groundTruthPoint.y);
	}
	error = sum / corners.size();
	return std::isfinite(error);
}

}  // namespace

// συγκέντρωση αποτελεσμάτων αξιολόγησης διαφορετικών queries
void HomographyTotals::merge(const HomographyTotals& other) {
	relevantPairs += other.relevantPairs;
	retrieved += other.retrieved;
	valid += other.valid;
	accurate += other.accurate;
	cornerErrorSum += other.cornerErrorSum;
}

// υπολογισμός ποσοστου εγγυρων εκτιμήσεων ομογραφίας
double HomographyTotals::validRate() const {
	if (relevantPairs > 0) {
		return static_cast<double>(valid) / relevantPairs;
	}
	return 0.0;
}

// υπολογισμός ποσοστού ακριβών εκτιμήσεων ομογραφίας
double HomographyTotals::accurateRate() const {
	if (relevantPairs > 0) {
		return static_cast<double>(accurate) / relevantPairs;
	}
	return 0.0;
}

// υπολογισμός μέσου corner error των εγγηρων εκτιμήσεων
double HomographyTotals::meanCornerError() const {
	if (valid > 0) {
		return cornerErrorSum / valid;
	}
	return std::numeric_limits<double>::quiet_NaN();
}

// αξιολόγηση των εκτιμώμενων ομογραφιών για ένα query
bool evaluateQueryHomographies(
    const std::vector<CandidateResult>& verified,
    const std::map<int, Eigen::Matrix3d>& groundTruth,
    const std::array<Point2, 4>& corners, HomographyTotals& totals,
    std::string& error) {
	// αερχικοποίηση των αποτελεσμάτων αξιολόγησης
	totals = HomographyTotals();
	totals.relevantPairs = static_cast<int>(groundTruth.size());
	// ελγχος δυνατότητας προβολής των γονιών με όλες τις ομογραφίες αναφοράς
	for (const auto& entry : groundTruth) {
		double ignored;
		if (!cornerError(entry.second, entry.second, corners, ignored)) {
			error =
			    "a ground-truth homography cannot project the image corners";
			return false;
		}
	}
	// αξιολόγηση των υποψείφιων εικόνων του αρχικού Top-C
	for (const CandidateResult& candidate : verified) {
		const auto gt = groundTruth.find(candidate.dbIndex);
		// παράλιψη εικόνων που δεν είναι σχετικές με το query
		if (gt == groundTruth.end()) {
			continue;
		}
		++totals.retrieved;
		// εελεγχος εγκυρότητας της εκτιμώμενης ομογραφίας
		if (!candidate.valid || candidate.inliers < 4) {
			continue;
		}
		// πολογισμός του σφάλματος προβολής των γονιών
		double cornerErr;
		if (!cornerError(candidate.H, gt->second, corners, cornerErr)) {
			continue;
		}
		++totals.valid;
		totals.cornerErrorSum += cornerErr;
		// έλεγχος ακρίβιας της εκτιμώμενης ομογραφίας
		if (cornerErr < kAccurateCornerErrorPx) {
			++totals.accurate;
		}
	}
	return true;
}