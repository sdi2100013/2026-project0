#include "kmeans.hpp"

#include <cmath>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>

namespace {

float sqDist(const float* a, const float* b, int d) {
	// Τετραγονικη αποσταση μεταξυ δυο σημειων
	float s = 0;
	for (int i = 0; i < d; ++i) {
		const float t = a[i] - b[i];
		s += t * t;
	}
	return s;
}

bool allFinite(const Matrix& m) {
	// Ελεγχος για NaN και infinite τιμες.
	for (float v : m.data)
		if (!std::isfinite(v)) return false;
	return true;
}

void checkData(const Matrix& data, int maxIter, double tol) {
	// Βασικος ελεγχος των παραμετρωων εισοδου
	if (data.rows < 1 || data.cols < 1 ||
	    data.data.size() != static_cast<size_t>(data.rows) * data.cols)
		throw std::invalid_argument(
		    "k-means: data must be a non-empty N x d matrix");
	if (maxIter < 1 || tol < 0)
		// Οι τιμες πρεπει να ειναι θετικες.
		throw std::invalid_argument(
		    "k-means: maxIter must be >= 1 and tol >= 0");
	if (!allFinite(data))
		// Απορριψη μη πεπερασμενων δεδομενων.
		throw std::invalid_argument(
		    "k-means: data contains NaN or infinite values");
}

// Βρισκει το κοντινοτερο centroid για κάθε σημειο και υπολογίζει το J
double assign(const Matrix& data, const Matrix& cent, std::vector<int>& labels,
              std::vector<float>& dist) {
	// Αναθεσει καθε σημειου στο κοντινοτερο centroid
	double J = 0;
	for (int i = 0; i < data.rows; ++i) {
		labels[i] = nearestCentroid(data.row(i), cent, &dist[i]);
		J += dist[i];  // Συνολικο κοστος της αναθεσης
	}
	return J;
}

// Πρωτο centroid τυχαία, τα επομενα με βάση την αποσταση απο τα ηδη επιλεγμενα
Matrix seedPlusPlus(const Matrix& data, int K, std::mt19937& rng) {
	// Αρχικοποιηση centroid με τον κανονα k-means++.
	const int N = data.rows, d = data.cols;
	Matrix cent(K, d);

	return cent;
}

}  // namespace

int nearestCentroid(const float* x, const Matrix& cent, float* dist) {
	// ψαχνει το centroid με την μικροτερη αποσταη
	int best = 0;

	return best;
}

KMeansResult kmeansLloyd(const Matrix& data, const Matrix& initialCentroids,
                         int maxIter, double tol) {
	// ελεγχος δεδομενων και αρχικων centroids.
	return res;
}

KMeansResult kmeans(const Matrix& data, int K, unsigned seed, int maxIter,
                    double tol) {
	// Δημιουργια αρχικων centroids με σταθερο sed.
	checkData(data, maxIter, tol);
	if (K <= 0) throw std::invalid_argument("k-means: K must be positive");
	if (K > data.rows)
		throw std::invalid_argument("k-means: K (" + std::to_string(K) +
		                            ") exceeds the number of points (" +
		                            std::to_string(data.rows) + ")");
	std::mt19937 rng(seed);
	return kmeansLloyd(data, seedPlusPlus(data, K, rng), maxIter, tol);
}