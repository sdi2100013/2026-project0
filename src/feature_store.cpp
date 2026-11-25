#include "feature_store.hpp"
#include "sift_extract.hpp"
#include "vocabulary.hpp"

#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <utility>



namespace fs = std::filesystem;

namespace {

//Αναγνωριστικό και έκδοση της μορφής του αρχείου αποθήκευσης.
constexpr char kMagic[8] = {'S', 'I', 'F', 'T', 'F', 'E', 'A', 'T'};
constexpr uint32_t kFeatureFormatVersion = 1;

//Διαστάσεις των SIFT descriptors
constexpr uint32_t kDim = kSiftDim;

//Μέγιστο αποδεκτό πλήθος εγγραφών.
constexpr uint64_t kMaxCount = 100000000;

//Μγεθος του checksum σε bytes.
constexpr size_t kChecksumSize = 16;

//ελεγχος αν όλοι οι χαρακτήρες ενός τμήματος του string
//οικανοποιούν μια συγκεκριμένη συνθήκη.
bool allOf(const std::string& s, size_t from, size_t to, bool (*predicate)(char)) {
    if (from >= to || to > s.size()) {
        return false;
    }
    for (size_t i = from; i < to; ++i) {
        if (!predicate(s[i])) {
            return false;
        }
    }
    return true;
}


//Έλεγχος αν ο χαρακτήρας είναι ψηφείο.
bool isDigit(char c) {
    return c >= '0' && c <= '9';
}

//Ελεγχος αν ο χαρακτήρας επιτρέπεται στο όνομα μιας ακολουθίας.
bool isNameChar(char c) {
    return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || isDigit(c) || c == '_';
}

}//Τέλος το namespace



//Eλεγχος εγκυρότητας του αναγνωριστικού μιας εικόνας.
static bool validImageKey(const ImageKey& key, std::string& error) {
    const std::string& id = key.id;
    //Έλεγχος αναγνωριστικού εικόνας HPatches.
    if (key.source == ImageSource::HPatches) {
        const size_t slash = id.find('/');
        if (slash == std::string::npos || id.size() != slash + 2 ||
            !allOf(id, 0, slash, isNameChar) || id.back() < '1' || id.back() > '6') {
            error = "invalid HPatches image id '" + id + "' (expected <sequence>/<1..6>)";
            return false;
        }
        return true;
    }
    //ελεγχος αναγνοριστικού εικόνας MIRFlickr.
    if (key.source == ImageSource::MirFlickr) {
        static const std::string prefix = "im";
        static const std::string suffix = ".jpg";
        if (id.size() <= prefix.size() + suffix.size() || id.compare(0, prefix.size(), prefix) != 0 ||
            id.compare(id.size() - suffix.size(), suffix.size(), suffix) != 0 || !allOf(id, prefix.size(), id.size() - suffix.size(), isDigit)) {
            error = "invalid MIRFlickr image id '" + id + "' (expected im<N>.jpg)";
            return false;
        }
        return true;
    }
    error = "unknown image source";
    return false;
}


//δημιουργία του ImageKey για την ηκόνα query μιας ακολουθίας HPatches.
ImageKey queryKey(const std::string& sequence) {
    return ImageKey{ImageSource::HPatches, sequence + "/1"};
}



//αναγνώριση του dataset προέλευσης και έλεγχος του αναγνωριστικού.
bool keyFromDbId(const std::string& dbId, ImageKey& key, std::string& error) {
    if (dbId.find('/') != std::string::npos) {
        key.source = ImageSource::HPatches;
    } else {
        key.source = ImageSource::MirFlickr;
    }
    key.id = dbId;
    return validImageKey(key, error);
}