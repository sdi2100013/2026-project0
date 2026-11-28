#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <vector>
#include <exception>

#include "args.hpp"
#include "database.hpp"
#include "evaluation.hpp"
#include "exact.hpp"
#include "feature_store.hpp"
#include "hpatches_gt.hpp"
#include "rerank.hpp"
#include "retrieval.hpp"
#include "vocabulary.hpp"

namespace {

using Clock = std::chrono::steady_clock;

//υπολογισμός του χρόνου που έχει περάσει σε milliseconds
double msSince(Clock::time_point start) {
	return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

//τρόποι κατάταξης των υποψηφιων ηκονων
enum Ranking {
	kBefore,
	kAfter,
	kByMatches,
	kRankings
};


//ονόματα των κατατάξεων για την εμφάνιση των αποτελεσμάτων
const char* const kRankingNames[kRankings] = {
	"Before", "After", "ByMatches"
};

//αποθήκευση των αποτελεσμάτων επαλήθευσης και αξιολογησης ενοσ query
struct QueryOutcome {
	//δείκτες των υποψήφιων εικόνων της βάσης από την αρχική ανάκτηση
	std::vector<int> candidates;
	//αποτελεσματα γεωμετρικής επαληθευσης των υποψηφίων
	std::vector<CandidateResult> results;
	//σειρά των υποψηφίων για κάθε τρόπο κατάταξης
	std::vector<int> order[kRankings];
	//μετρικες ανακτησης για κάθε κατάταξη
	QueryMetrics metrics[kRankings];
	//αποτελεσματα αξιολόγησης των εκτιμώμενων ομογραφιών
	HomographyTotals homography;
	//Χρόνοι ανάκτησης και γεωμετρικής επαλήθευσης
	double candidateMs = 0;
	double verifyMs = 0;
};

//μεταφορά των παραμέτρων εκτελεσης στις ριθμισεις του RANSAC
RansacParams ransacParams(const Config& cfg) {
	RansacParams params;
	params.tau = cfg.tau;
	params.maxIterations = cfg.T;
	params.p = cfg.p;
	params.adaptive = cfg.adaptive;
	return params;
}


//επιλογή των πρότων 10 εικόνων της καταταξης για την αξιολόγηση
std::vector<int> top10(const std::vector<int>& order, const std::vector<int>& candidates) {
	std::vector<int> top;
	for (size_t k = 0; k < order.size() &&
	     k < static_cast<size_t>(kTopK); ++k) {
		top.push_back(candidates[order[k]]);
	}
	return top;
}

// Ανάκτηση, γεωμετρική επαλήθευση και αξιολόγηση ενός query
bool processQuery(const Config& cfg, const SearchData& data, int q, const TopC& retrieve, FeatureStore& store,
                  QueryOutcome& out, std::string& error) {
	//δημιουργια του BoVW διανύσματος του συγκεκριμένου query
	Matrix query(1, data.queries.cols);
	for (int j = 0; j < query.cols; ++j) {
		query.row(0)[j] = data.queries.row(q)[j];
	}
	//ανάκτηση των C πρότων υποψήφιων εικόνων και μέτρηση χρόνου
	auto start = Clock::now();
	out.candidates = retrieve(query, cfg.C);
	out.candidateMs = msSince(start);
	//ααποδέσμευση των χαρακτηριστικών του προηγούμενου query από τη μνήμη
	store.clearMemory();
	//φορτωση ή εξαγωγή των SIFT χαρακτηριστικων του query
	const ImageFeatures* queryFeatures = nullptr;
	if (!store.get(queryKey(data.queryIds[q]), queryFeatures, error)) {
		return false;
	}
	//φόρτωση ή εξαγωγή των SIFT χαρακτηριστικών των υποψηφίων
	std::vector<const ImageFeatures*> candidateFeatures;
	candidateFeatures.reserve(out.candidates.size());
	for (int row : out.candidates) {
		ImageKey key;
		const ImageFeatures* features = nullptr;
		if (!keyFromDbId(data.dbIds[row], key, error)) {
			return false;
		}
		if (!store.get(key, features, error)) {
			return false;
		}
		candidateFeatures.push_back(features);
	}
    //προετοιμασια των παραμέτρων του RANSAC
    RansacParams ransac = ransacParams(cfg);
    //εναρξη μετρησης του χρόνου γεωμετρικής επαλιθευσης
    start = Clock::now();
    //Αρχικοποίηση των αποτελεσμάτων για κάθε υποψήφια εικόνα
    out.results.assign(out.candidates.size(), CandidateResult());
    //γεωμετρική επαλήθευση κάθε υποψήφιας εικόνας
    for (size_t i = 0; i < out.candidates.size(); ++i) {
	    std::vector<TwoNearest> nearest;
	    //ευ'ρεση των δύο κοντινότερων descriptors για κάθε descriptor του query
        if (!twoNearest(queryFeatures->descriptors, candidateFeatures[i]->descriptors, nearest, error)) {
		    return false;
        }
	    //καθορισμός seed για το συγκεκριμένο ζεύγος εικόνων
	    ransac.seed = ransacSeed(cfg.seed, q, out.candidates[i]);
        //εφαρμογή Lowe ratio test και εκτίμηση ομογραφίας με RANSAC
	    out.results[i] = verifyCandidate(*queryFeatures, *candidateFeatures[i], nearest, cfg.rho, ransac);
	    //αποθήκευση του δείκτη της υποψήφιας εικόνας στη βάση
	    out.results[i].dbIndex = out.candidates[i];
    }
    //διατείρηση της αρχικής σειράς ανάκτησης
    out.order[kBefore].resize(out.candidates.size());
    for (size_t i = 0; i < out.candidates.size(); ++i) {
	    out.order[kBefore][i] = static_cast<int>(i);
    }
    //επανακατάταξη με βαση τους τελικούς inliers του RANSAC
    out.order[kAfter] = rankByInliers(out.results);
    //επαανακατάταξη με βάση το πλήθος των αποδεκτών matches
    out.order[kByMatches] = rankByMatches(out.results);
    //λήξη μέτρησης του χρόνου γεωμετρικής επαλήθευσης
    out.verifyMs = msSince(start);
    //Αξιολόγηση των μετρικών ανάκτησης για κάθε τρόπο κατάταξης
    for (int r = 0; r < kRankings; ++r) {
	    out.metrics[r] = evaluateQuery(top10(out.order[r], out.candidates), data.relevant[q]);
    }
    //γορτωση των ομογραφιων αναφοράς όλων των σχετικών εικόνων
    std::map<int, Eigen::Matrix3d> groundTruth;
    for (int row : data.relevant[q]) {
	    const std::string path = groundTruthPath(cfg.hp, data.dbIds[row]);
        if (!readGroundTruth(path, groundTruth[row], error)) {
		    return false;
	    }
    }
    //υπολογισμός των συντταγμένων των τεσσάρων γωνιών του query
    const auto corners = imageCorners(queryFeatures->width, queryFeatures->height);
    //Αξιολογηση των εκτιμώμενων ομογραφιών των σχετικών υποψηφίων
    return evaluateQueryHomographies(out.results, groundTruth, corners, out.homography, error);
}


