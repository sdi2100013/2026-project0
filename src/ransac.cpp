#include "ransac.hpp"

#include <algorithm>
#include <cmath>
#include <random>

namespace {

// ελεγχος των παραμετρων πριν αρχισει ο ransac
bool validParams(const RansacParams& params, std::string& error) {
	// το tau πρεπει να ειναι αριθμος και θετικο
	if (!(std::isfinite(params.tau) && params.tau > 0)) {
		error = "tau must be finite and positive";
		// θελουμε τουλαχιστον μια δοκιμη
	} else if (params.maxIterations < 1) {
		error = "Tmax must be at least 1";
		// το p πρεπει να ειναι αναμεσα στο 0 και στο 1, αλλιως ο τυπος δεν
		// δουλευει
	} else if (!(std::isfinite(params.p) && params.p > 0 && params.p < 1)) {
		error = "p must be in (0, 1)";
	} else {
		return true;
	}
	return false;
}

// διαλεγει 4 διαφορετκους δεικτες χωρις να ξαναδιαλεξει το ιδιο
void pickFour(std::mt19937& rng, std::vector<int>& pool, int out[4]) {
	const int m = static_cast<int>(pool.size());
	for (int k = 0; k < 4; ++k) {
		// τυχαια θεση απο τη θεση k μεχρι το τελος (οι προηγουμενες εχουν
		// παρθει)
		std::uniform_int_distribution<int> pick(k, m - 1);
		// η τιμη που διαλεξαμε παει στη θεση k και δεν ξαναδιαλεγεται
		std::swap(pool[k], pool[pick(rng)]);
		out[k] = pool[k];
	}
}

// νεο οριο δοκιμων: min(currentLimit, ceil(log(1-p) / log(1-w^4))), w =
// bestInliers / M αν w = 0 το οριο μενει ιδιο, αν w = 1 γινεται 1, και ποτε δεν
// μεγαλωνει
int adaptiveLimit(int currentLimit, int bestInliers, int M, double p) {
	// χωρις inliers το οριο μενει ιδιο
	if (bestInliers <= 0) return currentLimit;
	// ποσοστο inliers της καλυτερης υποθεσης (το min το κραταει το πολυ στο 1)
	const double w = std::min(1.0, static_cast<double>(bestInliers) / M);
	// πιθανοτητα μια τετραδα να εχει μονο inliers
	const double allInliers = std::pow(w, 4);
	// αν w = 1 ολες οι τετραδες ειναι καθαρες και φτανει μια δοκιμη
	if (allInliers >= 1) return 1;
	// δοκιμες που χρειαζονται ωστε με πιθανοτητα p να βρουμε τετραδα χωρις
	// outliers η log1p δουλευει σωστα και οταν το allInliers ειναι πολυ μικρο
	const double needed = std::ceil(std::log(1 - p) / std::log1p(-allInliers));
	// το οριο μονο μικρενει, ποτε δεν μεγαλωνει
	if (!(needed < currentLimit)) return currentLimit;
	return std::max(1, static_cast<int>(needed));
}

// Οι δεικτες των ζευγων με σφαλμα μικρτερο απο tau (το απειρο δεν ειναι ποτε
// inlier)
std::vector<int> inlierIndices(const std::vector<double>& errors, double tau) {
	std::vector<int> indices;
	for (size_t i = 0; i < errors.size(); ++i) {
		if (errors[i] < tau) indices.push_back(static_cast<int>(i));
	}
	return indices;
}

// η μεση τιμη στη σειρα (median) των σφαλματων των inliers
double medianError(const std::vector<double>& errors,
                   const std::vector<int>& inliers) {
	// παιρνουμε μονο τα σφαλματα των inliers και τα βαζουμε σε σειρα
	std::vector<double> values;
	for (int i : inliers) values.push_back(errors[i]);
	std::sort(values.begin(), values.end());
	const size_t n = values.size();
	// μονο πληθος: το μεσαιο. ζυγο πληθος: ο μεσος των δυο μεσαιων
	return n % 2 == 1 ? values[n / 2]
	                  : 0.5 * (values[n / 2 - 1] + values[n / 2]);
}

}  // namespace

// ενα seed για καθ ζευγος (query, candidate) απο το βασικο seed
// το seed_seq ειναι ορισμενο απο το προτυπο, οποτε βγαινει το ιδιο παντου
// και δεν αλλαζει με τα αλλα ζευγη ουτε με το C
unsigned ransacSeed(unsigned baseSeed, int queryId, int candidateId) {
	std::seed_seq seq{baseSeed, static_cast<unsigned>(queryId),
	                  static_cast<unsigned>(candidateId)};
	unsigned seed = 0;
	seq.generate(&seed, &seed + 1);
	return seed;
}

// ransac με τελικο refit επιστρεφει true μονο οτν βρει σωστη ομογραφια
bool ransacHomography(const std::vector<PointCorrespondence>& correspondences,
                      const RansacParams& params, RansacResult& result,
                      std::string& error) {
	// αρχικα το αποτελεσμα ειναι αδειο, για να μην μεινουν παλιες τιμες αν
	// αποτυχει
	result = RansacResult();
	if (!validParams(params, error)) return false;
	// M: ποσα ζευγη περασαν το lowe test
	const int M = static_cast<int>(correspondences.size());
	// η ομογραφια θελει τουλαχιστον 4 ζευγη
	if (M < 4) {
		error = "fewer than 4 correspondences";
		return false;
	}
	// ολες οι συντεταγμενες πρεπει να ειναι κανονικοι αριθμοι
	for (const PointCorrespondence& pc : correspondences) {
		if (!std::isfinite(pc.p.x) || !std::isfinite(pc.p.y) ||
		    !std::isfinite(pc.c.x) || !std::isfinite(pc.c.y)) {
			error = "non-finite point coordinates";
			return false;
		}
	}

	// τυχαιοι αριθμοι με το seed του ζευγους (ετσι βγαινει παντα το ιδιο
	// αποτελεσμα)
	std::mt19937 rng(params.seed);
	// λιστα με τους δεικτες 0..M-1, απο εκει διαλεγουμε τις τετραδες
	std::vector<int> pool(M);
	for (int i = 0; i < M; ++i) pool[i] = i;
	// τα σφαλματα των M ζευγων για την υποθεση που δοκιμαζουμε
	std::vector<double> errors;
	// η καλυτερη υποθεση μεχρι τωρα: η H, ποσοι inliers και η μεση τιμη τους
	Eigen::Matrix3d bestH;
	int bestCount = 0;
	double bestMedian = std::numeric_limits<double>::infinity();
	// το οριο δοκιμων αρχιζει απο το Tmax και μπορει μονο να μικρυνει
	int limit = params.maxIterations;

	// 1. δοκιμες: τετραδα, ελεγχος αν ειναι κακη, DLT, inliers, καλυτερη
	// υποθεση
	for (int trial = 0; trial < limit; ++trial) {
		// τυχαια τετραδα ζευγων
		int pick[4];
		pickFour(rng, pool, pick);
		const std::array<PointCorrespondence, 4> quad = {
		    correspondences[pick[0]], correspondences[pick[1]],
		    correspondences[pick[2]], correspondences[pick[3]]};
		// οι κακες τετραδες μετραν ως δοκιμη αλλα δεν δινουν υποθεση
		if (isDegenerateSample(quad)) continue;
		// ομογραφια απο την τετραδα. αν αποτυχει, την προσπερναμε
		Eigen::Matrix3d H;
		std::string ignored;
		if (!estimateHomography(
		        std::vector<PointCorrespondence>(quad.begin(), quad.end()), H,
		        ignored) ||
		    !reprojectionErrors(H, correspondences, errors)) {
			continue;
		}
		// inliers της υποθεσης, μετρημενοι σε ολα τα M ζευγη
		const std::vector<int> inliers = inlierIndices(errors, params.tau);
		const int count = static_cast<int>(inliers.size());
		// τη μεση τιμη τη χρειαζομαστε μονο αν δεν εχουμε λιγοτερους απο τον
		// καλυτερο
		if (count == 0 || count < bestCount) continue;
		const double median = medianError(errors, inliers);
		// καλυτερη: περισσοτεροι inliers, ή ισοι και μικροτερη μεση τιμη
		if (count > bestCount || median < bestMedian) {
			bestH = H;
			bestCount = count;
			bestMedian = median;
			// με νεα καλυτερη υποθεση αλλαζει το οριο δοκιμων
			if (params.adaptive)
				limit = adaptiveLimit(limit, bestCount, M, params.p);
		}
	}
	// χωρις υποθεση με τουλχιστον 4 inliers η επαληθευση αποτυγχανει
	if (bestCount < 4) {
		error = "no valid hypothesis with at least 4 inliers";
		return false;
	}

	// 2. refit: νεα ομογραφια μονο απο τους inliers της καλυτερης υποθεσης
	reprojectionErrors(bestH, correspondences, errors);
	std::vector<PointCorrespondence> bestInliers;
	for (int i : inlierIndices(errors, params.tau))
		bestInliers.push_back(correspondences[i]);
	Eigen::Matrix3d finalH;
	if (!estimateHomography(bestInliers, finalH, error)) {
		error = "refit failed: " + error;
		return false;
	}

	// 3. τελικοι inliers: ξαναελεγχουμε ολα τα M ζευγη με την τελικη H
	// αυτος ο αριθμος μπαινει στην καταταξη, οχι ο προσωρινος της καλυτερης
	// υποθεσης
	reprojectionErrors(finalH, correspondences, errors);
	const std::vector<int> finalInliers = inlierIndices(errors, params.tau);
	if (finalInliers.size() < 4) {
		error = "refit failed: fewer than 4 final inliers";
		return false;
	}
	// επιτυχια: κραταμε την ομογραφια, τους inliers και τη μεση τιμη των
	// σφαλματων
	result.H = finalH;
	result.inliers = finalInliers;
	result.medianError = medianError(errors, finalInliers);
	result.valid = true;
	return true;
}