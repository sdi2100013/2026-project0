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



bool loadVocabulary(const std::string& path, const VocabMeta& expected, std::vector<float>& centroids, VocabMeta& meta, std::string& error) {
    const std::string bad = "vocabulary cache '" + path + "' ";
    if (!fs::exists(path)) {
        error = bad + "does not exist";
        return false;
    }
    try {
        std::ifstream in(path, std::ios::binary);
        if (!in.is_open()) {
            error = bad + "cannot be opened";
            return false;
        }
        int version = 0;
        in.read(reinterpret_cast<char*>(&version), sizeof(version));
        if(!in){
            error = bad + "is corrupted: cannot read format version";
            return false;
        }
        //Ανάγνωση των k,d,s από το αρχείο
        in.read(reinterpret_cast<char*>(&meta.K), sizeof(meta.K));
        in.read(reinterpret_cast<char*>(&meta.d), sizeof(meta.d));
        in.read(reinterpret_cast<char*>(&meta.S), sizeof(meta.S));
        if(!in){
            error = bad + "is corrupted: cannot read vocabulary dimensions";
            return false;
        }
        //Ανάγνωση seed
        in.read(reinterpret_cast<char*>(&meta.seed), sizeof(meta.seed));
        if(!in){
            error = bad + "is corrupted: cannot read seed";
            return false;
        }
        //Ανάγνωση του μήκους του splitHash
        std::uint32_t hashLength = 0;
        in.read(reinterpret_cast<char*>(&hashLength), sizeof(hashLength));
        if(!in){
            error = bad + "is corrupted: cannot read split hash length";
            return false;
        }
        if (hashLength > 1024) {
            error = bad + "is corrupted: invalid split hash length";
            return false;
        }
        meta.splitHash.resize(hashLength);
        in.read(meta.splitHash.data(), hashLength);
        if(!in){
            error = bad + "is corrupted: cannot read split hash";
            return false;
        }
        //Ανάγνωση ρυθμίσεων των SIFT
        std::uint32_t configLength = 0;
        in.read(reinterpret_cast<char*>(&configLength), sizeof(configLength));
        if(!in || configLength > 4096){
            error = bad + "is corrupted: invalid SIFT configuration length";
            return false;
        }
        meta.siftConfig.resize(configLength);
        in.read(meta.siftConfig.data(), configLength);
        if (!in) {
            error = bad + "is corrupted: cannot read SIFT configuration";
            return false;
        }
        //Ανάγωση πληροφοριών εκπαίδευσης
        in.read(reinterpret_cast<char*>(&meta.descriptors), sizeof(meta.descriptors));
        in.read(reinterpret_cast<char*>(&meta.iterations), sizeof(meta.iterations));
        in.read(reinterpret_cast<char*>(&meta.objectiveInitial), sizeof(meta.objectiveInitial));
        in.read(reinterpret_cast<char*>(&meta.objectiveFinal), sizeof(meta.objectiveFinal));
        if(!in){
            error = bad + "is corrupted: cannot read training metadata";
            return false;
        }
        auto mismatch = [&](const std::string& what, const std::string& have, const std::string& want) {
            error = bad + "is incompatible: " + what + " is " + have + " in the file but " + want + " was requested";
            return false;
        };
        if (version != kVocabFormatVersion){
            return mismatch("format version", std::to_string(version), std::to_string(kVocabFormatVersion));
        }
        if (meta.K != expected.K){
            return mismatch("K", std::to_string(meta.K), std::to_string(expected.K));
        }
        if (meta.d != expected.d){
            return mismatch("d", std::to_string(meta.d), std::to_string(expected.d));
        }
        if (meta.S != expected.S) {
            return mismatch("S", std::to_string(meta.S), std::to_string(expected.S));
        }
        if (meta.seed != expected.seed) {
            return mismatch("seed", std::to_string(meta.seed), std::to_string(expected.seed));
        }
        if (meta.splitHash != expected.splitHash) {
            return mismatch("split.txt hash", meta.splitHash, expected.splitHash);
        }
        if (meta.siftConfig != expected.siftConfig) {
            return mismatch("SIFT configuration", meta.siftConfig, expected.siftConfig);
        }
        //Ανάγνωση των centroidsS
        if(meta.K <= 0 || meta.d <=0){
            error = bad + "is corrupted: invalid centroid dimensions";
            return false;
        }
        const size_t centroidCount = static_cast<size_t>(meta.K) * meta.d;
        centroids.resize(centroidCount);
        in.read(reinterpret_cast<char*>(centroids.data()), centroidCount * sizeof(float));
        if (!in) {
            error = bad + "is corrupted: cannot read centroids";
            return false;
        }
        //Έλεγψος των τιμών centroids
        for(float value : centroids){
            if(!std::isfinite(value)){
                error = bad + "is corrupted: centroids contain NaN or infinite values";
                return false;
            }
        }
    } catch (const std::exception& e) {
        error = bad + "is corrupted: " + e.what();
        return false;
    }
    return true;
}