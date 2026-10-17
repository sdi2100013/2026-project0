#include "vocabulary.hpp"

#include <algorithm>
#include <filesystem>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>

#include "dataset.hpp"

namespace {
// Increment when SIFT, sampling or k-means settings change.
constexpr int formatVersion = 1;
constexpr int maxIterations = 100;
constexpr double epsilon = 1e-4;

void validateArguments(const std::vector<std::string>& images, int K, int S) {
	if (images.empty() || K <= 0 || S <= 0) {
		throw std::runtime_error(
		    "Vocabulary requires training images and positive K, S");
	}
}

// Include paths and file metadata to invalidate the cache when training inputs
// change.
std::vector<std::string> trainingManifest(
    const std::vector<std::string>& images) {
	std::vector<std::string> manifest;
	for (const auto& image : images) {
		const auto path = std::filesystem::canonical(image);
		std::ostringstream entry;
		entry << path.string() << '\t' << std::filesystem::file_size(path)
		      << '\t'
		      << std::filesystem::last_write_time(path)
		             .time_since_epoch()
		             .count();
		manifest.push_back(entry.str());
	}
	return manifest;
}
}  // namespace

cv::Mat trainVocabulary(const std::vector<std::string>& trainingImages, int K,
                        int S, int seed) {
	validateArguments(trainingImages, K, S);
	std::mt19937 generator(static_cast<unsigned int>(seed));
	cv::setRNGSeed(seed);
	cv::Mat centers;
	return centers;
}

cv::Mat loadOrTrainVocabulary(const std::string& file,
                              const std::vector<std::string>& trainingImages,
                              int K, int S, int seed) {
	validateArguments(trainingImages, K, S);

	cv::Mat centers = trainVocabulary(trainingImages, K, S, seed);

	return centers;
}
