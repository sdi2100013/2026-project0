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
	std::uniform_int_distribution<int> anyPoint(0, N - 1);
	std::vector<double> d2(N, std::numeric_limits<double>::max());

	for (int c = 0; c < K; ++c) {
		// πρωτο σημειο επιλεγεται χωρις βαρη.
		int idx = anyPoint(rng);
		if (c > 0) {
			double total = 0;
			for (double v : d2) total += v;
			if (total > 0) {  // αν υπαρχει απόσταση, γινεται weighted επιλογή
				// τα μακρινα σημεια εχουν μεγαλυτερη πιθανοτητα
				const double r =
				    std::uniform_real_distribution<double>(0.0, total)(rng);
				double acc = 0;
				idx = N - 1;
				for (int i = 0; i < N; ++i) {
					acc += d2[i];
					if (acc > r) {
						idx = i;
						break;
					}
				}
			}
		}
		for (int j = 0; j < d; ++j) cent.row(c)[j] = data.row(idx)[j];
		// ενημερωση της κοντινοτερης αποστασης καθε σημειου
		for (int i = 0; i < N; ++i) {
			const double v = sqDist(data.row(i), cent.row(c), d);
			if (v < d2[i]) d2[i] = v;
		}
	}
	return cent;
}

}  // namespace

int nearestCentroid(const float* x, const Matrix& cent, float* dist) {
	// ψαχνει το centroid με την μικροτερη αποσταη
	int best = 0;
	float bestD = sqDist(x, cent.row(0), cent.cols);
	for (int j = 1; j < cent.rows; ++j) {
		// σε ισοπαλια μενει το πρωτο centroid
		const float dj = sqDist(x, cent.row(j), cent.cols);
		if (dj < bestD) {
			bestD = dj;
			best = j;
		}
	}
	if (dist) *dist = bestD;
	return best;
}

KMeansResult kmeansLloyd(const Matrix& data, const Matrix& initialCentroids,
                         int maxIter, double tol) {
	// ελεγχος δεδομενων και αρχικων centroids.
	checkData(data, maxIter, tol);
	if (initialCentroids.cols != data.cols || initialCentroids.rows < 1 ||
	    initialCentroids.rows > data.rows ||
	    initialCentroids.data.size() !=
	        static_cast<size_t>(initialCentroids.rows) *
	            initialCentroids.cols ||
	    !allFinite(initialCentroids))
		throw std::invalid_argument(
		    "k-means: initial centroids must be finite, K x d with 1 <= K <= "
		    "N");

	const int N = data.rows, d = data.cols, K = initialCentroids.rows;
	KMeansResult res;
	res.centroids = initialCentroids;

	std::vector<int> labels(N), newLabels(N);
	std::vector<float> dist(N);
	// πρωτη αναθση πριν ξεκινησουν οι επαναληψεις
	res.objective.push_back(assign(data, res.centroids, labels, dist));

	for (int it = 1; it <= maxIter; ++it) {
		// μηδενισμος αθροισματων για τα νεα centroids
		// καθε centroid γινεται ο μεσος ορος του cluster του
		std::vector<double> sum(static_cast<size_t>(K) * d, 0.0);
		std::vector<int> count(K, 0);
		for (int i = 0; i < N; ++i) {
			// προσθικη του σημειου στο αντιστοιχο cluser
			const float* x = data.row(i);
			double* s = &sum[static_cast<size_t>(labels[i]) * d];
			for (int c = 0; c < d; ++c) s[c] += x[c];
			++count[labels[i]];
		}
		for (int j = 0; j < K; ++j) {
			// Υπολογισμος μεσου ορου για το cluser
			float* c = res.centroids.row(j);
			if (count[j] > 0) {
				for (int k = 0; k < d; ++k)
					c[k] = static_cast<float>(
					    sum[static_cast<size_t>(j) * d + k] / count[j]);
				continue;
			}
			int far = 0;  // αδειο cluster: παιρνει το πιο απομακρυσμένο σημειο
			for (int i = 1; i < N; ++i)
				if (dist[i] > dist[far]) far = i;
			if (dist[far] > 0) {
				// Το αδειο cluster παιρνει δυσκολο σημειο
				for (int k = 0; k < d; ++k) c[k] = data.row(far)[k];
				dist[far] = -1;
				++res.emptyRepaired;
			}
		}

		// νεα αναθεση σημειων με τα ενημερωμένα centroids
		const double prev = res.objective.back();
		const double J = assign(data, res.centroids, newLabels, dist);
		res.objective.push_back(J);
		res.iterations = it;
		const bool same = newLabels == labels;
		labels.swap(newLabels);
		// Τερματισμός αν δεν άλλαξαν οι αναθέσεις ή αν το κόστος μειώθηκε ελάχιστα
		if (same || (J <= prev && prev - J <= tol * prev)) {
    		res.converged = true;
    		break;
		}
	}
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