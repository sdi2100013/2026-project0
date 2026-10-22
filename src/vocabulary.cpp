#include "vocabulary.hpp"

//#include "kmeans.hpp"
//#include "sift_extract.hpp"

#include <cmath>
#include <cstdint>
#include <limits>
#include <exception>
#include <cstdint>
#include <vector>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iterator>




namespace fs = std::filesystem;

//*** FNV-1a!!!
//Υπολογισμός hash για δεδομένα στη μνήμη με τον αλγόριθμο FNV-1a
std::string hashBytes(const void* data, size_t size) {
    const unsigned char* bytes = static_cast<const unsigned char*>(data);
    unsigned long long hashValue = 1469598103934665603ULL;
    for (size_t i = 0; i < size; ++i){
        hashValue = (hashValue ^ bytes[i]) * 1099511628211ULL;
    }

    char buffer[17];
    std::snprintf(buffer, sizeof(buffer), "%016llx", hashValue);
    return buffer;
}

//Υπολογισμος hash από το περιεχόμενο ενός αρχείου.
bool fileHash(const std::string& path, std::string& hash, std::string& error) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        error = "cannot open '" + path + "'";
        return false;
    }
    const std::string content((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    hash = hashBytes(content.data(), content.size());
    return true;
}



//Αποθηκεύει τα centroids και τα metadata του vocabulary σε ένα αρχείο
bool saveVocabulary(const std::string& path, const std::vector<float>& centroids, const VocabMeta& meta, std::string& error) {
    if (meta.K <= 0 || meta.d <= 0 || centroids.size() != static_cast<size_t>(meta.K) * meta.d) {
        error = "centroid matrix does not match K x d of the metadata";
        return false;
    }
    for (float value : centroids) {
        if (!std::isfinite(value)) {
            error = "centroids contain NaN or infinite values";
            return false;
        }
    }
    if (meta.splitHash.size() > std::numeric_limits<std::uint32_t>::max() || meta.siftConfig.size() > std::numeric_limits<std::uint32_t>::max()) {
        error = "vocabulary metadata string is too long";
        return false;
    }
    //Δημιουργεία προσωρινού αρχείου για την αποθύκευση
    const std::string tmp = path + ".tmp";
    try {
        //Άνοιγμα προσωρινου αρχειου για εγγραφή
        std::ofstream out(tmp, std::ios::binary | std::ios::trunc);
        if (!out.is_open()) {
            error = "cannot write '" + tmp + "'";
            return false;
        }
        //Αποθύκευση των metadata του vocabulary
        const int version = kVocabFormatVersion;
        out.write(reinterpret_cast<const char*>(&version), sizeof(version));
        out.write(reinterpret_cast<const char*>(&meta.K), sizeof(meta.K));
        out.write(reinterpret_cast<const char*>(&meta.d), sizeof(meta.d));
        out.write(reinterpret_cast<const char*>(&meta.S), sizeof(meta.S));
        // Αποθήκευση του seed
        out.write(reinterpret_cast<const char*>(&meta.seed), sizeof(meta.seed));
        //Αποθύκευση του split hash
        const std::uint32_t hashLength = static_cast<std::uint32_t>(meta.splitHash.size());
        out.write(reinterpret_cast<const char*>(&hashLength), sizeof(hashLength));
        out.write(meta.splitHash.data(), hashLength);
        //Αποθύκευση των ρυθμίσεων SIFT
        const std::uint32_t configLength = static_cast<std::uint32_t>(meta.siftConfig.size());
        out.write(reinterpret_cast<const char*>(&configLength), sizeof(configLength));
        out.write(meta.siftConfig.data(), configLength);
        //Αποθύκευση πληροφοριών εκπαίδευσης
        out.write(reinterpret_cast<const char*>(&meta.descriptors), sizeof(meta.descriptors));
        out.write(reinterpret_cast<const char*>(&meta.iterations), sizeof(meta.iterations));
        out.write(reinterpret_cast<const char*>(&meta.objectiveInitial), sizeof(meta.objectiveInitial));
        out.write(reinterpret_cast<const char*>(&meta.objectiveFinal), sizeof(meta.objectiveFinal));
        //Αποθήκευση των centroids στο δυαδικό αρχείο
        out.write(reinterpret_cast<const char*> (centroids.data()), centroids.size() * sizeof(float));
        if(!out){
            error = "can't write vocabulary data to '" + tmp + "'";
            return false;
        }
        out.close();
        if(!out){
            error = "can't close '" + tmp + "'";
            return false;
        }
        //Μετονομασια του προσωινου αρχείου στο τελικό όνομα
        std::error_code renameError;
        fs::rename(tmp, path, renameError);
        if (renameError) {
            error = "cannot move '" + tmp + "' to '" + path + "': " + renameError.message();
            return false;
        }
    } catch (const std::exception& e) {
        //Διαχείριση σφαλμάτων κατά την αποθήκευση
        error = std::string("cannot save vocabulary: ") + e.what();
        return false;
    }
    return true;
}