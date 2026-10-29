#include "sift_extract.hpp"

#include <opencv2/features2d.hpp>
#include <opencv2/imgcodecs.hpp>

#include <algorithm>
#include <fstream>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <utility>


//Δημιουργεί ένα μοναδικό αναγνωριστικό με βάση τις παραμέτρους του SIFT
std::string siftConfigId() {
    std::ostringstream id;
    //Προσθήκη των παραμέτρων του SIFT στο αναγνωριστικό
    id << "sift_n" << kSiftFeatures << "_l" << kSiftOctaveLayers << "_c" << kSiftContrastThreshold << "_e" << kSiftEdgeThreshold << "_s" << kSiftSigma;
    return id.str();
}

//Διαβάζει εικόνα και εξάγει τους SIFT descriptors
bool extractSiftDescriptors(const std::string& path, cv::Mat& gray, std::vector<cv::KeyPoint>& keypoints, std::vector<float>& descriptors, std::string& error) {
    //Καθαρισμός των προηγούμενων αποτελεσμάτων
    keypoints.clear();
    descriptors.clear();

    try {
        gray = cv::imread(path, cv::IMREAD_GRAYSCALE);
        if (gray.empty()) {
            error = "cannot read image '" + path + "' (missing, unreadable or not a valid image)";
            return false;
        }
        //ΔΗΜΙΟΥΡΓΕΊΑ SIFT DETECTOR
        cv::Ptr<cv::SIFT> sift = cv::SIFT::create(kSiftFeatures, kSiftOctaveLayers, kSiftContrastThreshold, kSiftEdgeThreshold, kSiftSigma);
        cv::Mat desc;
        //ΕΝΤΟΠΊΖΕΙ ΧΑΡΑΚΤΗΡΙΣΤΙΚΑ ΣΗΜΕΙΑ ΤΗΣ ΕΙΚΟΝΑΣ
        //Υπολογύζιε έναν SIFT-descriptor 128 διαστάσεων για κάθε χαρακτηριστηκό σημείο
        sift->detectAndCompute(gray, cv::noArray(), keypoints, desc);
        //Μεταφορά των SIFT descriptors από cv::Mat σε std::vector<float>
        if(!desc.empty()){
            if(!desc.isContinuous()){
                error = "SIFT descriptors are not continuous in memory";
                return false;
            }
            descriptors.assign(desc.ptr<float>(0), desc.ptr<float>(0) + desc.total());
        }
    } catch (const cv::Exception& e) {
        error = "OpenCV error while processing '" + path + "': " + e.what();
        return false;
    }
    return true;
}

//Επιλέγει τυχαία μέχρι S descriptros
std::vector<float> sampleDescriptors(const std::vector<float>& descriptors, int S, std::mt19937& rng){
    //Έλεγχος ότι το S είναι θετικό
    if (S <= 0) {
        throw std::invalid_argument("sample size S must be positive");
    }
    //Έλεγχος ότι κάθε descriptor έχει 128 διαστάσεις
    if (descriptors.size() % kSiftDim != 0) {
        throw std::invalid_argument("descriptors size must be a multiple of 128");
    }
    //Υπολογισμός του πλήθους των descriptors
    const size_t N = descriptors.size() / kSiftDim;
    //Δημιουργία δεικτών για όλους τους descriptors
    std::vector<size_t> rows(N);
    std::iota(rows.begin(), rows.end(), 0);
    //Τυχαία αναδιάταξη των δεικτών
    std::shuffle(rows.begin(), rows.end(), rng);
    //Υπολογισμός του πλήθους των descriptors που θα επιλέξουμε
    const size_t n = std::min(N, static_cast<size_t>(S));
    //Δημιουργία vector για τους επιλεγμένους descriptors
    std::vector<float> sample(n * kSiftDim);
    //Αντιγραφή των τυχαία επιλεγμένων descriptors
    for (size_t i = 0; i < n; ++i) {
        const size_t source = rows[i] * kSiftDim;
        const size_t destination = i * kSiftDim;
        std::copy_n(descriptors.begin() + source, kSiftDim, sample.begin() + destination);
    }
    return sample;
}


//Διαβάζει το αρχείο split και αποθηκεύει τα sequences του ζητούμενου συνόλου
bool readSplitSequences(const std::string& splitPath, const std::string& set, std::vector<std::string>& names, std::string& error){
    //Καθαρισμός των προηγούμενων ονομάτων
    names.clear();
    //Άνοιγμα του αρχείου split
    std::ifstream in(splitPath);
    if (!in) {
        error = "cannot open split file '" + splitPath + "'";
        return false;
    }
    //Ανάγνωση του αρχείου γραμμή προς γραμμή
    std::string line;
    for (int lineNo = 1; std::getline(in, line); ++lineNo) {
        //Διαχωρισμός των στοιχείων της γραμμής
        std::istringstream ss(line);
        std::string name, lineSet, extra;
        //Παράλειψη των κενών γραμμών
        if (!(ss >> name)){
            continue;
        }
        //Έλεγχος ότι η γραμμή περιέχει έγκυρο όνομα και σύνολο
        if (!(ss >> lineSet) || (ss >> extra) || (lineSet != "train" && lineSet != "validation" && lineSet != "test")) {
            error = splitPath + ":" + std::to_string(lineNo) + ": invalid line '" + line + "'";
            return false;
        }
        //Αποθήκευση του sequence αν ανήκει στο ζητούμενο σύνολο
        if (lineSet == set){
            names.push_back(name);
        }
    }
    return true;
}


//Συγκεντρώνει τους SIFT descriptors από τις εικόνες εκπαίδευσης
bool collectTrainingDescriptors(const std::string& hpPath, const std::string& splitPath, int S, std::mt19937& rng, std::vector<float>& training, TrainingStats& stats, std::string& error){
    //Καθαρισμός προηγούμενων δεδομένων εκπαίδευσης
    training.clear();
    //Μηδενισμός των στατιστικών εκπαίδευσης
    stats = TrainingStats();
    if (S <= 0) {
        error = "sample size S must be positive";
        return false;
    }
    std::vector<std::string> sequences;
    if (!readSplitSequences(splitPath, "train", sequences, error)) return false;
    if (sequences.empty()) {
        error = "no training sequences found in '" + splitPath + "'";
        return false;
    }
    std::vector<float> samples;
    for (const std::string& seq : sequences) {
        ++stats.sequences;
        for (int k = 1; k <= 6; ++k) {
            const std::string path = hpPath + "/" + seq + "/" + std::to_string(k) + ".ppm";
            //Αποθύκευση της εικόνας και των SIFT descriptors
            cv::Mat gray;
            std::vector<float> desc;
            std::vector<cv::KeyPoint> kps;
            //Εξαγωγή των SIFT descriptors από την εικόνα
            if (!extractSiftDescriptors(path, gray, kps, desc, error)){
                return false;
            }
            //Τυχαία επιλογή μέχρι S descriptors από την εικόνα
            std::vector<float> sample = sampleDescriptors(desc, S, rng);
            //Ενημέρωση των στατιστικών για την εικόνα
            ++stats.images;
            if (desc.empty()) {
                ++stats.emptyImages;
            }
            //Υπολογισμός του πλήθους των descriptors πριν και μετά τη δειγματοληψία
            stats.before += desc.size() / kSiftDim;
            stats.after += sample.size() / kSiftDim;
            //Προσθήκη των επιλεγμένων descriptors στο συνολικό vector
            samples.insert(samples.end(), sample.begin(), sample.end());
        }
    }
    //Μεταφορά όλων των επιλεγμένων descriptors στα δεδομένα εκπαίδευσης
    training = std::move(samples);
    return true;
}