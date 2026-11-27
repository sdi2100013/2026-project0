#include "rerank.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <string>

namespace {

//Μετατροπή μιας αντιστοίχισης descriptors σε ζεύγος σημείων
PointCorrespondence toCorrespondence(const ImageFeatures& query, const ImageFeatures& candidate, const Match& match) {
	PointCorrespondence correspondence;
	correspondence.p = {query.x[match.q], query.y[match.q]};
	correspondence.c = {candidate.x[match.i], candidate.y[match.i]};
	return correspondence;
}

}

//γεωμετρική επαλήθευση μιας υποψήφιας εικόνας
CandidateResult verifyCandidate(const ImageFeatures& query, const ImageFeatures& candidate,
                                const std::vector<TwoNearest>& nearest, double rho, const RansacParams& ransac) {
	CandidateResult result;
	std::string error;
	//εφαρμογή του Lowe ratio test
	const std::vector<Match> matches = matchesFromNearest(nearest, rho);
	result.matches = static_cast<int>(matches.size());
	//διμιουργία γεωμετρικών αντιστοιχίσεων από τα αποδεκτά matches
	std::vector<PointCorrespondence> correspondences;
	correspondences.reserve(matches.size());
	for (const Match& m : matches) {
		correspondences.push_back(toCorrespondence(query, candidate, m));
	}
	//εκτίμηση ομογραφίας και εντοπισμός των τελικών inliers
	RansacResult ransacResult;
	if (ransacHomography(correspondences, ransac, ransacResult, error)) {
		result.valid = true;
		result.inliers = static_cast<int>(ransacResult.inliers.size());
		result.medianError = ransacResult.medianError;
		result.H = ransacResult.H;
	}
	return result;
}

namespace {

//σύγκριση υποψείφιων εικόνων με βάση το πλήθος των matches
bool betterByMatches(const CandidateResult& a, const CandidateResult& b) {
	return a.matches > b.matches;
}
//σύγγριση υποψήφιων ηκόνων με βάση τους τελικούς inliers
bool betterByInliers(const CandidateResult& a, const CandidateResult& b) {
	if (a.inliers != b.inliers) {
		return a.inliers > b.inliers;
	}
	//σε ισοβαθμία εξετάζουμε τη διάμεσο του σφάλματος επαναπροβολής
	const bool aHasMedian = !std::isnan(a.medianError);
	const bool bHasMedian = !std::isnan(b.medianError);
	//προτερεότητα στην εικόνα που διαθέτει τιμή διαμέσου
	if (aHasMedian != bHasMedian) {
		return aHasMedian;
	}
	//αν και οι δύο δεν διαθέτουν διάμεσο, παραμένουν ισόβαθμες
	if (!aHasMedian) {
		return false;
	}
	return a.medianError < b.medianError;
}
//ταξηνόμιση των δεικτών των υποψήφιων εικόνων
std::vector<int> rankBy(const std::vector<CandidateResult>& results, bool (*better)(const CandidateResult&, const CandidateResult&)) {
	std::vector<int> order(results.size());
	//αρχικοποίηση των δεικτών σύμφωνα με την αρχική σειρά ανάκτησης
	std::iota(order.begin(), order.end(), 0);
	//σταθερή ταξινόμηση ώστε να διατηρείται η αρχική σειρά στις ισοβαθμίες
	std::stable_sort(order.begin(), order.end(), [&](int i, int j) {
		return better(results[i], results[j]);
	});
	return order;
} 

}//τελοσ το Namspace

//επανακατάταξη με βάση το πλήθος των αποδεκτών matches
std::vector<int> rankByMatches(const std::vector<CandidateResult>& results) {
	return rankBy(results, betterByMatches);
}

//επανακατάταξη με βάση τους τελικούς inliers
std::vector<int> rankByInliers(const std::vector<CandidateResult>& results) {
	return rankBy(results, betterByInliers);
}