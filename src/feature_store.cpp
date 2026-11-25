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


//ανάγνωση διαδικών δεδομένων από buffer με έλεγχο ορίων
struct Reader {
    const std::vector<char>& buffer;
    size_t pos = 0;
    //ανάγνωση Ν bytes και μετακίνηση της θέσης ανάγνωσης
    bool take(void* dst, size_t n) {
        if (pos > buffer.size() || buffer.size() - pos < n) {
            return false;
        }
        std::memcpy(dst, buffer.data() + pos, n);
        pos += n;
        return true;
    }
    //ανάγνωση μιας τιμής ανάλογα με το μέγεθος του τύπου της
    template <class T>
    bool get(T& value) {
        return take(&value, sizeof(value));
    }
    //ανάγνωση string από το μήκος και τους χαρακτήρες του
    bool str(std::string& s) {
        uint32_t n = 0;
        if (!get(n) || n > 256 || buffer.size() - pos < n) {
            return false;
        }
        s.assign(buffer.data() + pos, n);
        pos += n;
        return true;
    }
};


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

//Ανάγνωση των αποθηκευμένων χαρακτηριστηκών μιας εικόνας.
static bool loadFeatures(const std::string& path, const ImageKey& key, uint64_t expectedImageBytes, ImageFeatures& out, std::string& error) {
    const std::string errorPrefix = "feature cache '" + path + "' ";
    //ανοιγμα του διαδικού αρχείου.
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        error = errorPrefix + "cannot be opened";
        return false;
    }
    //ανάγνωση ολόκληρου του αρχείου στο buffer.
    std::vector<char> buffer((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    Reader reader{buffer};
    // Μεταβλητές για την ανάγνωση της επικεφαλίδας.
    char magic[sizeof(kMagic)];
    uint32_t version = 0;
    uint32_t dim = 0;
    uint8_t source = 0;
    std::string id;
    std::string config;
    uint64_t imageBytes = 0;
    uint64_t count = 0;
    int32_t width = 0;
    int32_t height = 0;
    //έλεγχος του αναγνωριστικού της μορφής του αρχείου.
    if (!reader.take(magic, sizeof(magic)) || std::memcmp(magic, kMagic, sizeof(kMagic)) != 0) {
        error = errorPrefix + "is corrupted (not a feature cache)";
        return false;
    }
    //ελεγχος συμβατότητας της έκδοσης του αρχείου.
    if (!reader.get(version) || version != kFeatureFormatVersion) {
        error = errorPrefix + "has an incompatible format version";
        return false;
    }
    //Ανάγνωση των υπόλοιπων πεδίων της επικεφαλίδας.
    if (!reader.get(source) || !reader.str(id) || !reader.str(config) || !reader.get(imageBytes) ||
        !reader.get(width) || !reader.get(height) || !reader.get(dim) || !reader.get(count)) {
        error = errorPrefix + "is corrupted (truncated header)";
        return false;
    }
    //ελεγχος ότι τα χαρακτηριστηκά ανήκουν στη ζητούμενη εικόνα.
    if (source != static_cast<uint8_t>(key.source) || id != key.id) {
        error = errorPrefix + "belongs to another image ('" + id + "')";
        return false;
    }
    //ελεγχος συμβατότητας των ρυθμίσεων του SIFT.
    if (config != siftConfigId()) {
        error = errorPrefix + "was made with another SIFT configuration (" + config + ")";
        return false;
    }
    //έλεγχος αν έχει μεταβληθεί το μέγεθος του αρχείου εικόνας.
    if (imageBytes != expectedImageBytes) {
        error = errorPrefix + "does not match the image file (size changed)";
        return false;
    }
    //έλεγχος της διάστασης των descriptors.
    if (dim != kDim) {
        error = errorPrefix + "has descriptor dimension " + std::to_string(dim);
        return false;
    }
    //'εεγχος των διαστάσεων της ηκόνας και του πλήθους των keypoints.
    if (width <= 0 || height <= 0 || count > kMaxCount) {
        error = errorPrefix + "has invalid image size or keypoint count";
        return false;
    }
    //υπολογισμός του αναμενώμενου μεγέθους των χαρακτηριστηκών.
    const size_t n = static_cast<size_t>(count);
    const size_t payloadSize = n * (2 + kSiftDim) * sizeof(float);
    //ελεγχος του μεγέθους των υπόλιπων δεδομένων του αρχείου
    if (buffer.size() - reader.pos != payloadSize + kChecksumSize) {
        error = errorPrefix + "is corrupted (wrong length for " + std::to_string(count) + " keypoints)";
        return false;
    }
    //υπολογισμός του checksum των αποθηκευμένων χαρακτηριστικών
    const std::string checksum = hashBytes(buffer.data() + reader.pos, payloadSize);
    //σύγκριση του υπολογισμένου με το αποθηκευμένο checksum
    if (std::memcmp(checksum.data(), buffer.data() + reader.pos + payloadSize, kChecksumSize) != 0) {
        error = errorPrefix + "is corrupted (checksum mismatch)";
        return false;
    }
    //αρχικοποίηση της δομής με τα στοιχεία της εικόνας.
    ImageFeatures f;
    f.key = key;
    f.width = width;
    f.height = height;
    f.imageBytes = imageBytes;
    //δέσμευση χώρου για τις συντεταγμένες και τους descriptors.
    f.x.resize(n);
    f.y.resize(n);
    f.descriptors = Matrix(static_cast<int>(n), kSiftDim);
    //προσωρινός πίνακας ανάγνωσης των συντεταγμένων.
    std::vector<float> tmp(n);
    //ανάγνωση πρώτα των x και στη συνέχεια των y συντεταγμένων.
    for (std::vector<double>* coords : {&f.x, &f.y}) {
        if (n > 0 && !reader.take(tmp.data(), n * sizeof(float))) {
            error = errorPrefix + "has truncated keypoint coordinates";
            return false;
        }
        int limit;
        if (coords == &f.x) {
            limit = width;
        } else {
            limit = height;
        }
        // Έλεγχος εγκυρότητας κάθε συντεταγμένης.
        for (size_t i = 0; i < n; ++i) {
            if (!std::isfinite(tmp[i]) || tmp[i] < 0 || tmp[i] > limit) {
                error = errorPrefix + "has a keypoint outside the image";
                return false;
            }
            (*coords)[i] = tmp[i];
        }
    }
    //ανάγνωση των αποθηκευμένων SIFT descriptors.
    if (n > 0 && !reader.take(f.descriptors.data.data(), n * kSiftDim * sizeof(float))) {
        error = errorPrefix + "has truncated descriptors";
        return false;
    }
    //έλεγχος εγκυρότητας των τιμών των descriptors.
    for (float value : f.descriptors.data) {
        if (!std::isfinite(value)) {
            error = errorPrefix + "has non-finite descriptor values";
            return false;
        }
    }
    //μεταφορά των χαρακτηριστικών στη δομή εξόδου.
    out = std::move(f);
    return true;
}


//αρχικοποίηση των διαδρομών των datasets και του cache.
FeatureStore::FeatureStore(std::string hpRoot, std::string mirRoot, std::string cacheDir)
    : hp_(std::move(hpRoot)), mir_(std::move(mirRoot)), dir_(std::move(cacheDir)) {
}


//ευρεση της διαδρομης του αρχείου της ηκονας ανάλογα με το dataset.
std::string FeatureStore::imagePath(const ImageKey& key) const {
    if (key.source == ImageSource::HPatches) {
        return hp_ + "/" + key.id + ".ppm";
    } else {
        return mir_ + "/" + key.id;
    }
}

//ευρεση της διαδρομής του αρχείου αποθήκευσης των χαρακτηριστηκών.
std::string FeatureStore::cachePath(const ImageKey& key) const {
    const std::string base = dir_ + "/" + siftConfigId() + "/";
    if (key.source == ImageSource::HPatches) {
        return base + "hpatches/" + key.id + ".feat";
    } else {
        return base + "mirflickr/" + key.id + ".feat";
    }
}


//ανάκτηση των χαρακτηριστικών μιας εικόνας.
bool FeatureStore::get(const ImageKey& key, const ImageFeatures*& out, std::string& error) {
    out = nullptr;
    //'ελεγχος εγκυρότητας του αναγνωριστικού της εικόνας.
    if (!validImageKey(key, error)) {
        return false;
    }
    //δημιουργία μοναδικού κλιδιού για την αναζήτηση στη μνήμη.
    std::string memoryKey;
    if (key.source == ImageSource::HPatches) {
        memoryKey = "hp:" + key.id;
    } else {
        memoryKey = "mir:" + key.id;
    }
    //Αναζητηση των χαρακτηριστηκών στη μνήμη.
    auto it = memory_.find(memoryKey);
    if (it != memory_.end()) {
        out = &it->second;
        return true;
    }
    //εύρεση των διαδρομών της εικόνας και του αρχείου χαρακτηριστικών.
    const std::string image = imagePath(key);
    const std::string cache = cachePath(key);
    //ανάκτηση του μεγέθους του αρχείου της ηκόνας σε bytes.
    std::error_code ec;
    const uint64_t bytes = fs::file_size(image, ec);
    //έλεγχος για σφάλμα κατά την ανάκτηση του μεγέθους.
    if (ec) {
        error = "cannot read image '" + image + "' (missing or unreadable)";
        return false;
    }
    //αντικείμενο για την αποθήκευση των χαρακτηριστηκών της εικόνας.
    ImageFeatures f;
    std::string cacheError;
    //έλεγχος ύπαρξης και φώρτωση των χαρακτηριστικών από τον δίσκο.
    bool loaded = false;
    if (fs::exists(cache, ec)) {
        loaded = loadFeatures(cache, key, bytes, f, cacheError);
    }
    //αν δεν φορτώθηκαν έγκυρα χαρακτηριστικά, πραγματοποιείται εξαγωγή SIFT.
    if (!loaded) {
        //μεταβλητές για την εξαγωγή των SIFT χαρακτηριστικών.
        cv::Mat gray;
        std::vector<cv::KeyPoint> keypoints;
        std::vector<float> descriptors;
        //εξαγωγή των keypoints και των descriptors της εικόνας.
        if (!extractSiftDescriptors(image, gray, keypoints, descriptors, error)) {
            return false;
        }
        //΄ςλεγχος αντιστοιχίας του πλήθους των keypoints και των descriptors.
        if (descriptors.size() != keypoints.size() * kSiftDim) {
            error = "SIFT returned " + std::to_string(keypoints.size()) + " keypoints but " +
            std::to_string(descriptors.size()) + " descriptor values for '" + image + "'";
            return false;
        }
        //αποθήκευση του αναγνωριστικού, των διαστάσεων και του μεγέθους της εικόνας.
        f.key = key;
        f.width = gray.cols;
        f.height = gray.rows;
        f.imageBytes = bytes;
        //αποθήκευση των συντεταγμένων των keypoints.
        for (const cv::KeyPoint& k : keypoints) {
            f.x.push_back(k.pt.x);
            f.y.push_back(k.pt.y);
        }
        //Δημιουργία του πίνακα των descriptors.
        f.descriptors = Matrix(static_cast<int>(keypoints.size()), kSiftDim);
        //Μεταφορά των τημών των descriptors στον πίνακα.
        f.descriptors.data = std::move(descriptors);
        // Αποθήκευση των χαρακτηριστικών στον δίσκο.
        if (!saveFeatures(cache, f, error)) {
            return false;
        }
    }
    //μεταφορά των χαρακτηριστηκών στη μνήμη.
    memory_[memoryKey] = std::move(f);
    //Επιστροφή της διεύθινσης των αποθηκευμένων χαρακτηριστικών.
    out = &memory_[memoryKey];
    return true;
}