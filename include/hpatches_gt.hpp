#pragma once

#include <array>
#include <limits>
#include <map>
#include <string>
#include <vector>

#include "homography.hpp"
#include "rerank.hpp"

// αξιολόγηση των εκτιμομμενων ομογραφιών με τις ομογραφίες αναφοράς του
// HPatches. οι ομογραφίες αναφοράς χρησιμοποιούνται αποκληστικά για την
// αξιολόγηση.

// κατασκευή διαδρομής του αρχείου ομογραφίας αναφοράς
std::string groundTruthPath(const std::string& hpRoot, const std::string& dbId);

// ανάγνωση και έλεγχος εγκυρότητας της ομογραφίας αναφοράς
bool readGroundTruth(const std::string& path, Eigen::Matrix3d& H,
                     std::string& error);

// επιστροφή των τεσσάρων γωνιών της εικόνας
std::array<Point2, 4> imageCorners(double width, double height);

// μέγιστο αποκληστικό όριο σφάλματος για μια ακριβή εκτίμηση
constexpr double kAccurateCornerErrorPx = 5.0;

// συγκεντρωτικά αποτελέσματα αξιολόγησης ομογραφιών
struct HomographyTotals {
	int relevantPairs = 0;      // συνολικό πλήθος σχετικών ζευγών
	int retrieved = 0;          // σχετικές εικόνες που ανακτήθηκαν στο Top-C
	int valid = 0;              // πλήθος εγγηυρων εκτιμήσεων ομογραφίας
	int accurate = 0;           // εγκυρες εκτιμήσεις με corner error < 5 pixels
	double cornerErrorSum = 0;  // αθροισμα σφαλμάτων των έγγηρων εκτιμήσεων
	// συνένωση αποτελεσμάτων διαφορετικών queries
	void merge(const HomographyTotals& other);

	// Ποσοστό έγκυρων εκτιμήσεων επί όλων των σχετικών ζευγών
	double validRate() const;

	// Ποσοστό ακριβών εκτιμήσεων επί όλων των σχετικών ζευγών
	double accurateRate() const;

	// Μέσο σφάλμα προβολής γονιών των έγκυρων εκτιμήσεων
	// επιστρέφει NaN όταν δεν υπάρχουν έγκυρες εκτιμήσεις
	double meanCornerError() const;
};

// αξιολόγηση των εκτιμώμενων ομογραφιών για ένα query
// το groundTruth αντιστοιχίζει τους δείκτες των σχετικών ηκονων
// της βάσης στις πραγματικές ομογραφίες τους.
// Οι σχετικές εικόνες εκτός Top-C συνυπολογίζονται μόνο στον παρονομαστή.
bool evaluateQueryHomographies(
    const std::vector<CandidateResult>& verified,
    const std::map<int, Eigen::Matrix3d>& groundTruth,
    const std::array<Point2, 4>& corners, HomographyTotals& totals,
    std::string& error);