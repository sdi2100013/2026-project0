#pragma once

#include "args.hpp"
#include "vocabulary.hpp"
#include "kmeans.hpp"

#include <cstdint>
#include <string>
#include <vector>


//Δομή δεδομένων για τα queries και τη βάση αναζήτησης
struct SearchData {
    //Αναγνωριστικά των queries
    std::vector<std::string> queryIds;
    //Διανύσματα BoVW των queries
    Matrix queries;
    //Θέσεις των σχετικών εικόνων στη βάση για κάθε query
    std::vector<std::vector<int>> relevant;
    //Αναγνωριστικά των εικόνων της βάσης
    std::vector<std::string> dbIds;
    //Διανύσματα BoVW των εικόνων της βάσης
    Matrix database;
    //Πλήθος εικόνων από το HPatches
    int hpatchesImages = 0;
    //Πλήθος εικόνων MIRFlickr χωρίς SIFT descriptors
    int skippedDistractors = 0;
    //Πλήθος εικόνων που φορτώθηκαν από την cache ή υπολογίστηκαν
    int imagesFromCache = 0;
    int imagesComputed = 0;
};


//Δομή για την αποθήκευση των BoVW διανυσμάτων στην cache
struct BovwCache {
    //Πληροφορίες για κάθε εικόνα της cache
    struct Entry {
        //Μέγεθος του αρχείου εικόνας σε bytes
        uint64_t fileSize = 0;
        //Θέση του BoVW διανύσματος στον πίνακα (-1 αν δεν υπάρχουν SIFT descriptors)
        int row = -1;
    };
    //Διαδρομή του αρχείου cache
    std::string path;
    //Όνομα της συλλογής εικόνων
    std::string label;
    //Παράμετροι του vocabulary
    int K = 0;
    int S = 0;
    unsigned seed = 0;
    //Ρυθμίσεις εξαγωγής SIFT
    std::string siftConfig;
    //Hashes για τον έλεγχο συμβατότητας της cache
    std::string vocabHash;
    std::string splitHash;
    //Πληροφορίες για κάθε εικόνα
    std::vector<Entry> entries;
    //Πίνακας με τα αποθηκευμένα BoVW διανύσματα
    Matrix vectors;
    //Πλήθος νέων εγγραφών που δεν έχουν αποθηκευτεί
    int unsaved = 0;
};

//Συνολικό πλήθος εικόνων του MIRFlickr
constexpr int kMirflickrImages = 25000;


//Αποθηκεύει την cache των BoVW διανυσμάτων σε δυαδικό αρχείο
bool saveBovwCache(BovwCache& cache, std::string& error);


//Φορτώνει την cache των BoVW διανυσμάτων από δυαδικό αρχείο
bool loadBovwCache(BovwCache& cache, std::string& error);