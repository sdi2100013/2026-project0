#include <iostream>
#include <fstream>

#include "args.hpp"
#include "sift_extract.hpp"
#include "vocabulary.hpp"
#include "database.hpp"
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

    if (sample.size() != 3 * kSiftDim) {
        std::cerr << "SIFT: FAIL\n";
        return 1;
    }
    std::cout << "SIFT: PASS\n";

    //TEST 1: Εκπαίδευση vocabulary
    int K = 8;
    int S = 5;
    unsigned seed = 42;
    std::string cachePath = "data/test_vocab.bin";

    std::vector<float> vocab, loadedVocab;
    VocabInfo info, loadedInfo;

    if (!trainOrLoadVocabulary(hpPath, splitPath, K, S, seed,
                               cachePath, vocab, info, error)) {
        std::cerr << "Vocabulary: FAIL - " << error << '\n';
        return 1;
    }

    //TEST 2: Φόρτωση vocabulary από cache
    error.clear();
    if (!trainOrLoadVocabulary(hpPath, splitPath, K, S, seed,
                               cachePath, loadedVocab, loadedInfo, error) ||
        !loadedInfo.fromCache || loadedVocab != vocab) {
        std::cerr << "Vocabulary: FAIL - " << error << '\n';
        return 1;
    }
    std::cout << "Vocabulary: PASS\n";

    //TEST 3: Αποθήκευση και φόρτωση BoVW cache
    BovwCache testCache;
    testCache.path = "data/test_bovw.bin";
    testCache.label = "test";
    testCache.K = 3;
    testCache.S = 5;
    testCache.seed = 42;
    testCache.siftConfig = siftConfigId();

    //Δοκιμαστικά hashes των 16 bytes
    testCache.vocabHash = std::string(16, 'a');
    testCache.splitHash = std::string(16, 'b');

    //Δημιουργία πίνακα για ένα BoVW διάνυσμα
    testCache.vectors = Matrix(1, 3);
    testCache.vectors.data = {0.2f, 0.3f, 0.5f};

    //Προσθήκη μιας εγγραφής εικόνας
    BovwCache::Entry entry;
    entry.fileSize = 1234;
    entry.row = 0;
    testCache.entries.push_back(entry);

    BovwCache loadedCache = testCache;
    loadedCache.entries.clear();
    loadedCache.vectors = Matrix(0, 3);

    error.clear();
    if (!saveBovwCache(testCache, error) ||
        !loadBovwCache(loadedCache, error)) {
        std::cerr << "BoVW cache: FAIL - " << error << '\n';
        return 1;
    }

    //Έλεγχος ότι τα δεδομένα της cache δεν άλλαξαν
    if (loadedCache.vectors.rows != testCache.vectors.rows ||
        loadedCache.vectors.cols != testCache.vectors.cols ||
        loadedCache.vectors.data != testCache.vectors.data ||
        loadedCache.entries.size() != testCache.entries.size() ||
        loadedCache.entries[0].fileSize != testCache.entries[0].fileSize ||
        loadedCache.entries[0].row != testCache.entries[0].row) {
        std::cerr << "BoVW cache: FAIL\n";
        return 1;
    }
    std::cout << "BoVW cache: PASS\n";

    //TEST 7: Αρχικοποίηση των πινάκων αναζήτησης
    SearchData testData;
    initIfNew(testData, 3);

    //Έλεγχος των διαστάσεων των πινάκων
    if (testData.database.rows != 0 || testData.database.cols != 3 ||
        testData.queries.rows != 0 || testData.queries.cols != 3) {
        std::cerr << "InitIfNew: FAIL\n";
        return 1;
    }
    std::cout << "InitIfNew: PASS\n";
    
    //TEST KMEANS
    Matrix points(4, 2);
    points.data = {0, 0, 0, 2, 10, 10, 10, 12};

    auto km = kmeans(points, 2, 42);
    auto km2 = kmeans(points, 2, 42);

    //Έλεγχος διαστάσεων, μείωσης κόστους και σταθερού seed
    bool ok = km.centroids.rows == 2 && km.centroids.cols == 2 && km.objective.size() >= 2 && km.objective.back() <= km.objective.front() && km.centroids.data == km2.centroids.data;

    //Έλεγχος ότι τα δύο clusters διαχωρίζονται σωστά
    int a = nearestCentroid(points.row(0), km.centroids);
    int b = nearestCentroid(points.row(1), km.centroids);
    int c = nearestCentroid(points.row(2), km.centroids);
    int d = nearestCentroid(points.row(3), km.centroids);
    ok = ok && a == b && c == d && a != c;

    if (!ok) {
        std::cerr << "KMeans: FAIL\n";
        return 1;
    }
    std::cout << "KMeans: PASS\n";


    std::cout << "\nALL TESTS PASSED\n";
    return 0;

    //4. Execute queries

    //5. Evaluate results

    //6. wrhite output
}
