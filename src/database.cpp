#include "database.hpp"

#include <cstdint>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>


namespace {
namespace fs = std::filesystem;

//Αναγνωριστικό του δυαδικού αρχείου cache
constexpr char kCacheMagic[8] = {'B', 'O', 'V', 'W', 'C', 'A', 'C', 'H'};
//Έκδοση της μορφής του αρχείου cache
constexpr uint32_t kCacheVersion = 2;

//Εγγραφή μιας τιμής σε δυαδικό αρχείο
template <class T>
void put(std::ofstream& out, const T& v) {
    out.write(reinterpret_cast<const char*>(&v), sizeof v);
}

//Ανάγνωση μιας τιμής από δυαδικό αρχείο
template <class T>
bool get(std::ifstream& in, T& v) {
    return static_cast<bool>(in.read(reinterpret_cast<char*>(&v), sizeof v));
}
} // namespace


//Αποθηκεύει την cache των BoVW διανυσμάτων σε δυαδικό αρχείο
bool saveBovwCache(BovwCache& cache, std::string& error) {
    //Αν δεν υπάρχει διαδρομή, δεν χρειάζεται αποθήκευση
    if (cache.path.empty()){
        return true;
    }
    //Προσωρινό αρχείο για ασφαλή αποθήκευση
    const std::string tmp = cache.path + ".tmp";
    std::ofstream out(tmp, std::ios::binary);
    if (!out) {
    error = "cannot write '" + tmp + "'";
        return false;
    }
    //Εγγραφή του αναγνωριστικού και της έκδοσης της cache
    out.write(kCacheMagic, sizeof kCacheMagic);
    put(out, kCacheVersion);
    //Εγγραφή των παραμέτρων του vocabulary
    put(out, static_cast<int32_t>(cache.K));
    put(out, static_cast<int32_t>(cache.S));
    put(out, static_cast<uint32_t>(cache.seed));
    //Εγγραφή των hashes του vocabulary και του split
    out.write(cache.vocabHash.data(), 16);
    out.write(cache.splitHash.data(), 16);
    //Εγγραφή του ονόματος της συλλογής
    put(out, static_cast<uint32_t>(cache.label.size()));
    out.write(cache.label.data(), cache.label.size());
    //Εγγραφή των ρυθμίσεων SIFT
    put(out, static_cast<uint32_t>(cache.siftConfig.size()));
    out.write(cache.siftConfig.data(), cache.siftConfig.size());
    //Εγγραφή του πλήθους των εικόνων της cache
    put(out, static_cast<uint64_t>(cache.entries.size()));
    //Εγγραφή των πληροφοριών κάθε εικόνας
    for (const BovwCache::Entry& e : cache.entries) {
        put(out, e.fileSize);
        put(out, static_cast<uint8_t>(e.row >= 0));
        if (e.row >= 0){
            //Έλεγχος εγκυρότητας της θέσης του BoVW διανύσματος
            if (cache.vectors.cols != cache.K || static_cast<size_t>(e.row) >= static_cast<size_t>(cache.vectors.rows)) {
                error = "invalid BoVW cache vector";
                return false;
            }
            out.write(reinterpret_cast<const char*>(cache.vectors.data.data() + static_cast<size_t>(e.row) * cache.K), sizeof(float) * cache.K);
        }
    }
    //Ολοκλήρωση της εγγραφής στο αρχείο
    out.flush();
    if (!out) {
        error = "error while writing '" + tmp + "'";
        return false;
    }
    out.close();
    if (!out) {
        error = "error while closing '" + tmp + "'";
        return false;
    }
    //Μετονομασία του προσωρινού αρχείου στην τελική διαδρομή
    std::error_code ec;
    fs::rename(tmp, cache.path, ec);
    if (ec) {
        error = "cannot move '" + tmp + "' to '" + cache.path + "': " + ec.message();
        return false;
    }
    cache.unsaved = 0;
    return true;
}

//Φορτώνει την cache των BoVW διανυσμάτων από δυαδικό αρχείο
bool loadBovwCache(BovwCache& cache, std::string& error) {
    //Καθαρισμός των προηγούμενων δεδομένων της cache
    cache.entries.clear();
    cache.vectors = Matrix(0, cache.K);
    cache.unsaved = 0;
    //Έλεγχος αν υπάρχει το αρχείο της cache
    std::error_code ec;
    if (cache.path.empty() || !fs::exists(cache.path, ec)){
        return true;
    }
    const std::string bad = "BoVW cache '" + cache.path + "' ";
    std::ifstream in(cache.path, std::ios::binary);
    //Μεταβλητές για την ανάγνωση του αναγνωριστικού και της έκδοσης
    char magic[8];
    uint32_t version = 0, seed = 0, labelSize = 0;
    int32_t K = 0, S = 0;
    uint64_t count = 0;
    char vocabHash[16], splitHash[16];
    //Έλεγχος αν το αρχείο είναι έγκυρη BoVW cache
    if (!in.read(magic, 8) || std::memcmp(magic, kCacheMagic, 8) != 0 || !get(in, version)) {
        error = bad + "is corrupted (not a BoVW cache)";
        return false;
    }
    //Έλεγχος συμβατότητας της έκδοσης της cache
    if (version != kCacheVersion) {
        error = bad + "is incompatible: format version is " + std::to_string(version) + " but " + std::to_string(kCacheVersion) + " is required";
        return false;
    }
    //Ανάγνωση των παραμέτρων και των hashes της cache
    if (!get(in, K) || !get(in, S) || !get(in, seed) || !in.read(vocabHash, 16) || !in.read(splitHash, 16) || !get(in, labelSize) || labelSize > 256) {
        error = bad + "is corrupted (truncated header)";
        return false;
    }
    std::string label(labelSize, '\0'), sift;
    uint32_t siftSize = 0;
    //Ανάγνωση του ονόματος, των ρυθμίσεων SIFT και του πλήθους εγγραφών
    if (!in.read(&label[0], labelSize) || !get(in, siftSize) || siftSize > 256 || !(sift.assign(siftSize, '\0'), in.read(&sift[0], siftSize)) || !get(in, count)) {
        error = bad + "is corrupted (truncated header)";
        return false;
    }
    //Δημιουργία μηνύματος σφάλματος όταν οι παράμετροι της cache δεν ταιριάζουν
    auto mismatch = [&](const std::string& what, const std::string& have, const std::string& want) {
        error = bad + "is incompatible: " + what + " is " + have + " in the file but " + want + " was requested";
        return false;
    };
    if (label != cache.label){
        return mismatch("collection", label, cache.label);
    }
    if (sift != cache.siftConfig){
        return mismatch("SIFT configuration", sift, cache.siftConfig);
    }
    if (K != cache.K){
        return mismatch("K", std::to_string(K), std::to_string(cache.K));
    }
    if (S != cache.S){
        return mismatch("S", std::to_string(S), std::to_string(cache.S));
    }
    if (seed != cache.seed){
        return mismatch("seed", std::to_string(seed), std::to_string(cache.seed));
    }
    if (std::string(vocabHash, 16) != cache.vocabHash){
        return mismatch("vocabulary fingerprint", std::string(vocabHash, 16), cache.vocabHash);
    }
    if (std::string(splitHash, 16) != cache.splitHash){
        return mismatch("split.txt hash", std::string(splitHash, 16), cache.splitHash);
    }
    std::vector<float> row(K);
    //Ανάγνωση των εγγραφών της cache
    for (uint64_t i = 0; i < count; ++i) {
        BovwCache::Entry e;
        uint8_t valid = 0;
        //Ανάγνωση του μεγέθους της εικόνας και της ένδειξης ύπαρξης BoVW
        if (!get(in, e.fileSize) || !get(in, valid) || valid > 1) {
            error = bad + "is corrupted (truncated or invalid entry " + std::to_string(i) + ")";
            return false;
        }
        //Ανάγνωση του BoVW διανύσματος αν υπάρχει
        if (valid){
            if (!in.read(reinterpret_cast<char*>(row.data()), sizeof(float) * K)) {
                error = bad + "is corrupted (truncated vector " + std::to_string(i) + ")";
                return false;
            }
            //Έλεγχος για NaN ή άπειρες τιμές στο BoVW διάνυσμα
            for (float value : row) {
                if (!std::isfinite(value)) {
                    error = bad + "is corrupted (NaN or infinite values in entry " + std::to_string(i) + ")";
                    return false;
                }
            }
            //Αποθήκευση της θέσης του νέου BoVW διανύσματος
            e.row = cache.vectors.rows;
            //Προσθήκη του διανύσματος στον πίνακα της cache
            cache.vectors.data.insert(cache.vectors.data.end(), row.begin(), row.end());
            ++cache.vectors.rows;
        }
        //Προσθήκη της εγγραφής στη λίστα της cache
        cache.entries.push_back(e);
    }
    //Έλεγχος αν υπάρχουν επιπλέον δεδομένα στο αρχείο
    if (in.peek() != std::ifstream::traits_type::eof()) {
        error = bad + "is corrupted (unexpected data after the last entry)";
        return false;
    }
    return true;
}
