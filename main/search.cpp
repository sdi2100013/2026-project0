#include <iostream>

#include "args.hpp"
#include "sift_extract.hpp"
#include "vocabulary.hpp"
#include "exact.hpp"
#include "lsh.hpp"
#include "hypercube.hpp"
#include "ivfflat.hpp"
#include "ivfpq.hpp"
#include "evaluation.hpp"


int main(int argc, char* argv[]){
    //1. Parse command-line arguments
    Config cfg;
    std::string error;
    //Έλεγχοσ παραμέτρων
    if (!parseArgs(std::vector<std::string>(argv + 1, argv + argc), cfg, error)) {
        std::cerr << "error: " << error << "\n";
        return 2;
    }
    //Έλεγχος ότι επιλεχθηκε σωστά ο αλγοριθμος αναζήτησης
    if (cfg.method != "exact" && cfg.method != "lsh" &&
        cfg.method != "hypercube" && cfg.method != "ivfflat" &&
        cfg.method != "ivfpq") {
        std::cerr << "error: -" << cfg.method
                  << " is not supported yet\n";
        return 1;
    }
    std::mt19937 rng(42);
    //Διαδρομή του dataset
    std::string hpPath = "data/hpatches-sequences-release";
    //Διαδρομή του αρχείου split
    std::string splitPath = "split.txt";
    //TEST SIFT SAMPLING
    std::vector<float> descriptors(10 * kSiftDim, 1.0f);
    std::vector<float> sample = sampleDescriptors(descriptors, 3, rng);
    std::cout << "SIFT sampling: "
              << (sample.size() == 3 * kSiftDim ? "PASS" : "FAIL") << '\n';

    std::cout << "SIFT config: " << siftConfigId() << '\n';
    std::vector<float> empty;
    std::cout << "Empty descriptors: " << (sampleDescriptors(empty, 3, rng).empty() ? "PASS" : "FAIL") << '\n';
    //TEST 1: Εκπαίδευση vocabulary
    int K = 8;
    int S = 5;
    unsigned seed = 42;
    std::string cachePath = "data/test_vocab.bin";
    std::vector<float> vocab;
    VocabInfo info;
    std::cout << "\n=== TEST 1: TRAIN VOCABULARY ===\n";
    if (!trainOrLoadVocabulary(
            hpPath, splitPath, K, S, seed,
            cachePath, vocab, info, error)) {

        std::cerr << "FAIL: " << error << '\n';
        return 1;
    }
    std::cout << "PASS\n";
    std::cout << "Centroids: " << vocab.size() / kSiftDim << '\n';
    std::cout << "Descriptors: " << info.meta.descriptors << '\n';
    std::cout << "Iterations: " << info.meta.iterations << '\n';
    std::cout << "Objective initial: " << info.meta.objectiveInitial << '\n';
    std::cout << "Objective final: " << info.meta.objectiveFinal << '\n';
    std::cout << "From cache: " << info.fromCache << '\n';
    std::cout << "Time: " << info.seconds << " seconds\n";
    //TEST 2: Φόρτωση vocabulary από cache
    std::cout << "\n=== TEST 2: LOAD CACHE ===\n";
    std::vector<float> loadedVocab;
    VocabInfo loadedInfo;
    error.clear();
    if (!trainOrLoadVocabulary(
            hpPath, splitPath, K, S, seed,
            cachePath, loadedVocab, loadedInfo, error)) {

        std::cerr << "FAIL: " << error << '\n';
        return 1;
    }
    if (!loadedInfo.fromCache) {
        std::cerr << "FAIL: Vocabulary was not loaded from cache\n";
        return 1;
    }
    if (loadedVocab != vocab) {
        std::cerr << "FAIL: Loaded vocabulary differs\n";
        return 1;
    }
    std::cout << "PASS\n";
    std::cout << "From cache: " << loadedInfo.fromCache << '\n';
    std::cout << "Centroids: " << loadedVocab.size() / kSiftDim << '\n';

    std::cout << "\nALL TESTS PASSED\n";

    return 0;

    //4. Execute queries

    //5. Evaluate results

    //6. wrhite output
}