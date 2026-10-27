#include "bovw.hpp"

#include <stdexcept>
#include <vector>

#include "kmeans.hpp"

cv::Mat computeBovw(const cv::Mat& descriptors, const cv::Mat& vocabulary) {
	// Ελεγχος τυπων και διαστασεων των πινακων
	if (descriptors.type() != CV_32F || vocabulary.type() != CV_32F ||
	    descriptors.dims != 2 || vocabulary.dims != 2)
		throw std::invalid_argument(
		    "BoVW: descriptors and vocabulary must be 2-D CV_32F matrices");
	if (vocabulary.rows < 1 || descriptors.cols != vocabulary.cols)
		throw std::invalid_argument("BoVW: descriptors have " +
		                            std::to_string(descriptors.cols) +
		                            " columns but the vocabulary has " +
		                            std::to_string(vocabulary.cols));
	if (descriptors.rows < 1)
		throw std::invalid_argument("BoVW: the image has no SIFT descriptors");
	// αποριψη μη πεπερασμενων τιμων.
	if (!cv::checkRange(descriptors) || !cv::checkRange(vocabulary))
		throw std::invalid_argument(
		    "BoVW: NaN or infinite values in descriptors or vocabulary");

	std::vector<int> counts(vocabulary.rows, 0);
	// Μετριση του centroid που αντιστοιχει σε κθε descriptor
	for (int i = 0; i < descriptors.rows; ++i)
		++counts[nearestCentroid(descriptors.ptr<float>(i), vocabulary)];

	// κανονικοποιηση των μετρησεων σε histogram
	cv::Mat bovw(1, vocabulary.rows, CV_32F);
	for (int j = 0; j < vocabulary.rows; ++j)
		bovw.at<float>(0, j) = static_cast<float>(
		    static_cast<double>(counts[j]) / descriptors.rows);
	return bovw;
}
