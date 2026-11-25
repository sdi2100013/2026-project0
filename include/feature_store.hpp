#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "kmeans.hpp"  // Το χρειαζόμαστε για την δομη Matrix

//FeatureStore: διατείρηση και ανάκτηση των αυθεντικών SIFT keypoints και
//descriptors ανά ηκόνα (για τη γεωμετρική επαλήθευση της 2ης εργασίας).
//Ανεξάρτητο από τα BoVW/IVF/vocabulary caches. Η εξαγωγή γίνεται με την
//υπάρχουσα extractSiftDescriptors (ίδιες παράμετροι SIFT, siftConfigId()).
//Η εξαγωγή γίνεται μόνο για τις εικόνες που ζητούνται.

//Προσδιορισμός του dataset προέλευσης της εικόνας.
enum class ImageSource : uint8_t {HPatches = 1, MirFlickr = 2};

//Ταυτότητα ηκόνας: dataset προέλευσης και αναγνωριστικό ηκόνας.
//HPatches: π.χ. v_bird/2, MIRFlickr: π.χ. im123.jpg.
struct ImageKey {
    ImageSource source = ImageSource::HPatches; 
    std::string id;
};


//Δημιουργία του αναγνωριστικού της ηκόνας query μιας ακολουθίας HPatches.
ImageKey queryKey(const std::string& sequence);


//Έλεγχος και μετατροπή του αναγνοριστικού μιας εικόνας της βάσης σε ImageKey.
bool keyFromDbId(const std::string& dbId, ImageKey& key, std::string& error);


//Τα χαρακτηριστικά μιας εικόνας: (x_i, y_i, descriptor_i), i = 0..N-1, με τη
//σειρά που τα επέστρεψε το SIFT. Οι συντεταγμένες είναι σε pixels της
//αρχικής εικόνας (KeyPoint::pt, float ακριβώς διευρυμένο σε double).
struct ImageFeatures {
    ImageKey key;
    //διαστάσεις της αρχικής ηκόνας.
    int width = 0;
    int height = 0;
    //μέγεθος του αρχείου της ηκόνας σε bytes.
    uint64_t imageBytes = 0;
    //συντεταγμένες των keypoints σε pixels της αρχικής ηκόνας.
    std::vector<double> x;
    std::vector<double> y;
    //πινακας N x 128 με τους αρχικούς SIFT descriptors.
    Matrix descriptors;
    //Επιστροφή του πλήθους των keypoints.
    int size() const {
        return static_cast<int>(x.size());
    }
};