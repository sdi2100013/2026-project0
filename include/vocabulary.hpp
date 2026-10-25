#pragma once

#include <opencv2/core.hpp>

#include <string>
#include <vector>

//Αποθηκεύει τις πληροφορίες που περιγράφουν ένα εκπαιδευμένο visual vocabulary
struct VocabMeta {
    int K = 0;                //Πλήθος οπτικών λέξεων
    int d = 0;                //Διάσταση descriptors
    int S = 0;                //Μέγιστο πλήρθος descriptors ανα εικόνα στην εκπαίδευση
    unsigned seed = 0;
    std::string splitHash;    //Hash του split.txt που χρησημοποιήθηκε στην εκπαίδευση
    std::string siftConfig;   //Ρυθμσίσεις SIFT των descriptors που χρησημοποιήθηκαν
    //Πληροφορίες που δεν συγκρίνονται κατά τη φόρτωση
    int descriptors = 0;      //Πλήθος descriptors που δόθηκαν στο kmeans
    int iterations = 0;
    double objectiveInitial = 0;
    double objectiveFinal = 0;
};

//Έκδοση της μορφής αποθήκευσης του vocabulry cache
constexpr int kVocabFormatVersion = 3;

//Αποθηκεύει τα centroids (K x d, CV_32F) και τα metadata στο path
//χρησιμοποιώντας προσωρινό αρχείο μέσω cv::FileStorage
//σε περίπτωση αποτυχίας επιστρέφει false και συμπληρώνει το eror
bool saveVocabulary(const std::string& path, const std::vector<float>& centroids, const VocabMeta& meta, std::string& error);



//Φορτώνει το vocabulary από το path και ελέγχει τη συμβατότητα
//της έκδοσης, των παραμέτρων K, d, S, seed και του split hash
//ελέγχει επίσης τις διαστάσεις και την εγκυρότητα των centroids
//σε οποιοδήποτε πρόβλημα επιστρέφει false
bool loadVocabulary(const std::string& path, const VocabMeta& expected, std::vector<float>& centroids, VocabMeta& meta, std::string& error);


//Υπολογισμός hash FNV-1a 64-bit (16 δεκαεξαδικά ψηφία)
//για δεδομένα σε bytes ή για το περιεχόμενο ενός αρχείου.
std::string hashBytes(const void* data, size_t size);
bool fileHash(const std::string& path, std::string& hash, std::string& error);