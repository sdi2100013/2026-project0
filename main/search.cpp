#include <exception>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "config.hpp"
#include "dataset.hpp"
#include "vocabulary.hpp"

int main(int argc, char* argv[]) {
	try {
		const Config config = parseArguments(argc, argv);
		// φορτωση αρχειου εκπαιδευσης
		const auto sequences = loadSplit(config.splitFile, "train");
		if (sequences.size() != 70) {
			throw std::runtime_error(
			    "Expected 70 training sequences in split file");
		}
		// συλλογη φωτογραφιων
		std::vector<std::string> trainingImages;
		for (const auto& sequence : sequences) {
			const auto paths = allhpatchesImPaths(config.hpPath, sequence);
			trainingImages.insert(trainingImages.end(), paths.begin(),
			                      paths.end());
		}

		// Κατασκευή αρχειου λεξιλογίου
		const std::string vocabularyFile =
		    config.vocabularyFile.empty()
		        ? "cache/vocabulary_K" + std::to_string(config.vocab) + "_S" +
		              std::to_string(config.S) + "_seed" +
		              std::to_string(config.seed) + ".yml"
		        : config.vocabularyFile;
		const bool cached = std::filesystem::exists(vocabularyFile);
		std::cout << (cached ? "Loading visual vocabulary: "
		                     : "Training visual vocabulary: ")
		          << vocabularyFile << std::endl;
		const cv::Mat centers =
		    loadOrTrainVocabulary(vocabularyFile, trainingImages, config.vocab,
			                      config.S, config.seed);
		std::cout << "Visual vocabulary ready: " << centers.rows
		          << " centers x " << centers.cols << " dimensions.\n";

		// Next stage: encode images with these centers, build the index and
		// search.
		std::cout << "Image histograms and search are not implemented yet.\n";
		return 0;
	} catch (const std::exception& error) {
		std::cerr << "Error: " << error.what() << '\n';
		return 1;
	}
}
