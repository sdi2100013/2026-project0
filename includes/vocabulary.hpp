#pragma once

#include <opencv2/core.hpp>
#include <string>
#include <vector>

// Οι γραμμες αντισοιχουν σε λεξεις. καθε γραμμη περιέχει κεντρα 128 διαστασεων
cv::Mat trainVocabulary(const std::vector<std::string>& trainingImages, int K,
                        int S, int seed);

// φορτωση συμβατού λεξιλογιου ή εκπαιδευση ή αποθηκευσε αν λειπει.
cv::Mat loadOrTrainVocabulary(const std::string& file,
                              const std::vector<std::string>& trainingImages,
                              int K, int S, int seed);
