#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "args.hpp"
#include "kmeans.hpp"
#include "vocabulary.hpp"

// Δομή δεδομένων για τα queries και τη βάση αναζήτησης
struct SearchData {
	// Αναγνωριστικά των queries
	std::vector<std::string> queryIds;
	// Διανύσματα BoVW των queries
	Matrix queries;
	// Θέσεις των σχετικών εικόνων στη βάση για κάθε query
	std::vector<std::vector<int>> relevant;
	// Αναγνωριστικά των εικόνων της βάσης
	std::vector<std::string> dbIds;
	// Διανύσματα BoVW των εικόνων της βάσης
	Matrix database;
	// Πλήθος εικόνων από το HPatches
	int hpatchesImages = 0;
	// Πλήθος εικόνων MIRFlickr χωρίς SIFT descriptors
	int skippedDistractors = 0;
	// Πλήθος εικόνων που φορτώθηκαν από την cache ή υπολογίστηκαν
	int imagesFromCache = 0;
	int imagesComputed = 0;
};

// Δομή για την αποθήκευση των BoVW διανυσμάτων στην cache
struct BovwCache {
	// Πληροφορίες για κάθε εικόνα της cache
	struct Entry {
		// Μέγεθος του αρχείου εικόνας σε bytes
		uint64_t fileSize = 0;
		// Θέση του BoVW διανύσματος στον πίνακα (-1 αν δεν υπάρχουν SIFT
		// descriptors)
		int row = -1;
	};
	// Διαδρομή του αρχείου cache
	std::string path;
	// Όνομα της συλλογής εικόνων
	std::string label;
	// Παράμετροι του vocabulary
	int K = 0;
	int S = 0;
	unsigned seed = 0;
	// Ρυθμίσεις εξαγωγής SIFT
	std::string siftConfig;
	// Hashes για τον έλεγχο συμβατότητας της cache
	std::string vocabHash;
	std::string splitHash;
	// Πληροφορίες για κάθε εικόνα
	std::vector<Entry> entries;
	// Πίνακας με τα αποθηκευμένα BoVW διανύσματα
	Matrix vectors;
	// Πλήθος νέων εγγραφών που δεν έχουν αποθηκευτεί
	int unsaved = 0;
};

// Συνολικό πλήθος εικόνων του MIRFlickr
constexpr int kMirflickrImages = 25000;

// Αποθηκεύει την cache των BoVW διανυσμάτων σε δυαδικό αρχείο
bool saveBovwCache(BovwCache& cache, std::string& error);

// Φορτώνει την cache των BoVW διανυσμάτων από δυαδικό αρχείο
bool loadBovwCache(BovwCache& cache, std::string& error);

// Αρχικοποίηση των πινάκων αναζήτησης αν είναι άδειοι
void initIfNew(SearchData& data, int K);

// Υπολογισμός BoVW για μία εικόνα
bool imageBovw(const std::string& path, const Matrix& vocabulary, Matrix& bovw,
               bool& empty, std::string& error);

// Appends the queries and relevant images of every `set` ("validation" or
// "test") sequence of the split file. An image without SIFT descriptors is an
// error. With a `cache`, images already in it are not run through SIFT again
// and new ones are added to it.
bool addHpatches(const std::string& hpPath, const std::string& splitPath,
                 const std::string& set, const Matrix& vocabulary,
                 SearchData& data, std::string& error,
                 BovwCache* cache = nullptr);

// Προσθήκη των πρώτων D έγκυρων εικόνων από το MIRFLICKR (im1.jpg, im2.jpg,
// ...) Οι εικόνες εξετάζονται με αύξουσα σειρά, μέχρι το πολύ total εικόνες
// Εικόνες χωρίς SIFT descriptors παραλείπονται και εξετάζεται η επόμενη
// Αν μια εικόνα δεν διαβάζεται ή δεν βρεθούν D έγκυρες εικόνες, επιστρέφεται
// σφάλμα Η cache αποθηκεύεται κάθε 2000 νέες εικόνες για να μη χάνεται η
// πρόοδος
bool addDistractors(const std::string& mirPath, int D, int total,
                    const Matrix& vocabulary, SearchData& data,
                    std::string& error, BovwCache* cache = nullptr);

// Προετοιμασία όλων των δεδομένων που χρειάζονται οι μέθοδοι αναζήτησης
// Φόρτωση του vocabulary από την cache ή εκπαίδευση και αποθήκευσή του
// Δημιουργία των queries και της βάσης δεδομένων σύμφωνα με τα cfg.set και
// cfg.D Τα BoVW διανύσματα του HPatches και του MIRFlickr αποθηκεύονται στην
// cache στον φάκελο data/
bool prepareSearchData(const Config& cfg, SearchData& data,
                       VocabInfo& vocabInfo, std::string& error);