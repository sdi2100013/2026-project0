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


//Προσθήκη της διαδικής αναπαράστασης μιας τιμής στο buffer.
template <class T>
void put(std::vector<char>& buffer, const T& value) {
    const char* ptr = reinterpret_cast<const char*>(&value);
    buffer.insert(buffer.end(), ptr, ptr + sizeof(value));
}

//Αποθήκευση του μήκους ενός string και στη συνέχεια των χαρακτήρων του.
void putStr(std::vector<char>& buffer, const std::string& s) {
    put(buffer, static_cast<uint32_t>(s.size()));
    buffer.insert(buffer.end(), s.begin(), s.end());
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


// Αποθήκευση των χαρακτηριστικών μιας εικόνας σε δυαδικό αρχείο.
static bool saveFeatures(const std::string& path, const ImageFeatures& f, std::string& error) {
    const size_t n = f.x.size();
    //ελεγχος συνέπειας του πλήθους των keypoints και descriptors.
    if (n > kMaxCount || f.y.size() != n || f.descriptors.rows != static_cast<int>(n) ||
        f.descriptors.cols != kSiftDim || f.descriptors.data.size() != n * kSiftDim) {
        error = "inconsistent feature arrays for '" + f.key.id + "'";
        return false;
    }
    //δημιουργία του buffer και αποθήκευση της επικεφαλήδας.
    std::vector<char> buffer;
    buffer.insert(buffer.end(), kMagic, kMagic + sizeof(kMagic));
    put(buffer, kFeatureFormatVersion);
    put(buffer, static_cast<uint8_t>(f.key.source));
    putStr(buffer, f.key.id);
    putStr(buffer, siftConfigId());
    put(buffer, f.imageBytes);
    put(buffer, static_cast<int32_t>(f.width));
    put(buffer, static_cast<int32_t>(f.height));
    put(buffer, kDim);
    put(buffer, static_cast<uint64_t>(n));
    //αρχική θέση των δεδομένων των χαρακτηριστικών.
    const size_t payloadStart = buffer.size();
    //αποθήκευση των συντεταγμένων των keypoints.
    for (double value : f.x) {
        put(buffer, static_cast<float>(value));
    }
    for (double value : f.y) {
        put(buffer, static_cast<float>(value));
    }
    // Αποθήκευση των SIFT descriptors.
    if (n > 0) {
        const char* ptr = reinterpret_cast<const char*>(f.descriptors.data.data());
        buffer.insert(buffer.end(), ptr, ptr + n * kSiftDim * sizeof(float));
    }
    //υπολογισμός και αποθήκευση του checksum των χαρακτηριστηκών.
    const std::string checksum = hashBytes(buffer.data() + payloadStart, buffer.size() - payloadStart);
    buffer.insert(buffer.end(), checksum.begin(), checksum.end());
    //δημιουργία του καταλόγου αποθήκευσης, εάν δεν υπάρχει.
    std::error_code ec;
    const fs::path p(path);
    if (p.has_parent_path()) {
        fs::create_directories(p.parent_path(), ec);
        if (ec) {
            error = "cannot create directory '" + p.parent_path().string() + "': " + ec.message();
            return false;
        }
    }
    // Εγγραφή των δεδομένων σε προσωρινό δυαδικό αρχείο.
    const std::string tmp = path + ".tmp";
    {
        std::ofstream out(tmp, std::ios::binary);
        if (!out) {
            error = "cannot write '" + tmp + "'";
            return false;
        }
        out.write(buffer.data(), static_cast<std::streamsize>(buffer.size()));
        out.flush();
        if (!out) {
            error = "error while writing '" + tmp + "'";
            return false;
        }
    }
    // Μετονομασία του προσωρινού αρχείου στο τελικό αρχείο.
    fs::rename(tmp, path, ec);
    if (ec) {
        error = "cannot move '" + tmp + "' to '" + path + "': " + ec.message();
        return false;
    }
    return true;
}

