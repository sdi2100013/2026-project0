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

