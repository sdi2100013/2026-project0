#include "bovw.hpp"

#include <cmath>
#include <stdexcept>
#include <vector>

#include "kmeans.hpp"

Matrix computeBovw(const Matrix& descriptors, const Matrix& vocabulary) {
	// Ελεγχος διαστασεων των πινακων
	if (vocabulary.rows < 1 || descriptors.cols != vocabulary.cols) {
		throw std::invalid_argument("BoVW: incompatible vocabulary dimensions");
	}
	if (descriptors.rows < 1) {
		throw std::invalid_argument("BoVW: the image has no SIFT descriptors");
	}
	// Ελεγχος για NaN και infinity στους descriptors
	for (float value : descriptors.data) {
		if (!std::isfinite(value)) {
			throw std::invalid_argument(
			    "BoVW: NaN or infinite values in descriptors or vocabulary");
		}
	}
	// Ελεγχος για NaN και infinity στο vocabulary
	for (float value : vocabulary.data) {
		if (!std::isfinite(value)) {
			throw std::invalid_argument(
			    "BoVW: NaN or infinite values in descriptors or vocabulary");
		}
	}
	std::vector<int> counts(vocabulary.rows, 0);
	// Μετριση του centroid που αντιστοιχει σε κθε descriptor
	for (int i = 0; i < descriptors.rows; ++i) {
		++counts[nearestCentroid(descriptors.row(i), vocabulary)];
	}
	// κανονικοποιηση των μετρησεων σε histogram
	Matrix bovw(1, vocabulary.rows);
	for (int j = 0; j < vocabulary.rows; ++j) {
		bovw.row(0)[j] = static_cast<float>(static_cast<double>(counts[j]) /
		                                    descriptors.rows);
	}
	return bovw;
}