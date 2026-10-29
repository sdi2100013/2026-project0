#include "database.hpp"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <utility>

#include "bovw.hpp"
#include "sift_extract.hpp"

namespace {
namespace fs = std::filesystem;

// Αναγνωριστικό του δυαδικού αρχείου cache
constexpr char kCacheMagic[8] = {'B', 'O', 'V', 'W', 'C', 'A', 'C', 'H'};
// Έκδοση της μορφής του αρχείου cache
constexpr uint32_t kCacheVersion = 2;

// Εγγραφή μιας τιμής σε δυαδικό αρχείο
template <class T>
void put(std::ofstream& out, const T& v) {
	out.write(reinterpret_cast<const char*>(&v), sizeof v);
}

// Ανάγνωση μιας τιμής από δυαδικό αρχείο
template <class T>
bool get(std::ifstream& in, T& v) {
	return static_cast<bool>(in.read(reinterpret_cast<char*>(&v), sizeof v));
}
}  // namespace

// Αποθηκεύει την cache των BoVW διανυσμάτων σε δυαδικό αρχείο
bool saveBovwCache(BovwCache& cache, std::string& error) {
	// Αν δεν υπάρχει διαδρομή, δεν χρειάζεται αποθήκευση
	if (cache.path.empty()) {
		return true;
	}
	// Προσωρινό αρχείο για ασφαλή αποθήκευση
	const std::string tmp = cache.path + ".tmp";
	std::ofstream out(tmp, std::ios::binary);
	if (!out) {
		error = "cannot write '" + tmp + "'";
		return false;
	}
	// Εγγραφή του αναγνωριστικού και της έκδοσης της cache
	out.write(kCacheMagic, sizeof kCacheMagic);
	put(out, kCacheVersion);
	// Εγγραφή των παραμέτρων του vocabulary
	put(out, static_cast<int32_t>(cache.K));
	put(out, static_cast<int32_t>(cache.S));
	put(out, static_cast<uint32_t>(cache.seed));
	// Εγγραφή των hashes του vocabulary και του split
	out.write(cache.vocabHash.data(), 16);
	out.write(cache.splitHash.data(), 16);
	// Εγγραφή του ονόματος της συλλογής
	put(out, static_cast<uint32_t>(cache.label.size()));
	out.write(cache.label.data(), cache.label.size());
	// Εγγραφή των ρυθμίσεων SIFT
	put(out, static_cast<uint32_t>(cache.siftConfig.size()));
	out.write(cache.siftConfig.data(), cache.siftConfig.size());
	// Εγγραφή του πλήθους των εικόνων της cache
	put(out, static_cast<uint64_t>(cache.entries.size()));
	// Εγγραφή των πληροφοριών κάθε εικόνας
	for (const BovwCache::Entry& e : cache.entries) {
		put(out, e.fileSize);
		put(out, static_cast<uint8_t>(e.row >= 0));
		if (e.row >= 0) {
			// Έλεγχος εγκυρότητας της θέσης του BoVW διανύσματος
			if (cache.vectors.cols != cache.K ||
			    static_cast<size_t>(e.row) >=
			        static_cast<size_t>(cache.vectors.rows)) {
				error = "invalid BoVW cache vector";
				return false;
			}
			out.write(reinterpret_cast<const char*>(cache.vectors.data.data() +
			                                        static_cast<size_t>(e.row) *
			                                            cache.K),
			          sizeof(float) * cache.K);
		}
	}
	// Ολοκλήρωση της εγγραφής στο αρχείο
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
	// Μετονομασία του προσωρινού αρχείου στην τελική διαδρομή
	std::error_code ec;
	fs::rename(tmp, cache.path, ec);
	if (ec) {
		error = "cannot move '" + tmp + "' to '" + cache.path +
		        "': " + ec.message();
		return false;
	}
	cache.unsaved = 0;
	return true;
}

// Φορτώνει την cache των BoVW διανυσμάτων από δυαδικό αρχείο
bool loadBovwCache(BovwCache& cache, std::string& error) {
	// Καθαρισμός των προηγούμενων δεδομένων της cache
	cache.entries.clear();
	cache.vectors = Matrix(0, cache.K);
	cache.unsaved = 0;
	// Έλεγχος αν υπάρχει το αρχείο της cache
	std::error_code ec;
	if (cache.path.empty() || !fs::exists(cache.path, ec)) {
		return true;
	}
	const std::string bad = "BoVW cache '" + cache.path + "' ";
	std::ifstream in(cache.path, std::ios::binary);
	// Μεταβλητές για την ανάγνωση του αναγνωριστικού και της έκδοσης
	char magic[8];
	uint32_t version = 0, seed = 0, labelSize = 0;
	int32_t K = 0, S = 0;
	uint64_t count = 0;
	char vocabHash[16], splitHash[16];
	// Έλεγχος αν το αρχείο είναι έγκυρη BoVW cache
	if (!in.read(magic, 8) || std::memcmp(magic, kCacheMagic, 8) != 0 ||
	    !get(in, version)) {
		error = bad + "is corrupted (not a BoVW cache)";
		return false;
	}
	// Έλεγχος συμβατότητας της έκδοσης της cache
	if (version != kCacheVersion) {
		error = bad + "is incompatible: format version is " +
		        std::to_string(version) + " but " +
		        std::to_string(kCacheVersion) + " is required";
		return false;
	}
	// Ανάγνωση των παραμέτρων και των hashes της cache
	if (!get(in, K) || !get(in, S) || !get(in, seed) ||
	    !in.read(vocabHash, 16) || !in.read(splitHash, 16) ||
	    !get(in, labelSize) || labelSize > 256) {
		error = bad + "is corrupted (truncated header)";
		return false;
	}
	std::string label(labelSize, '\0'), sift;
	uint32_t siftSize = 0;
	// Ανάγνωση του ονόματος, των ρυθμίσεων SIFT και του πλήθους εγγραφών
	if (!in.read(&label[0], labelSize) || !get(in, siftSize) ||
	    siftSize > 256 ||
	    !(sift.assign(siftSize, '\0'), in.read(&sift[0], siftSize)) ||
	    !get(in, count)) {
		error = bad + "is corrupted (truncated header)";
		return false;
	}
	// Δημιουργία μηνύματος σφάλματος όταν οι παράμετροι της cache δεν
	// ταιριάζουν
	auto mismatch = [&](const std::string& what, const std::string& have,
	                    const std::string& want) {
		error = bad + "is incompatible: " + what + " is " + have +
		        " in the file but " + want + " was requested";
		return false;
	};
	if (label != cache.label) {
		return mismatch("collection", label, cache.label);
	}
	if (sift != cache.siftConfig) {
		return mismatch("SIFT configuration", sift, cache.siftConfig);
	}
	if (K != cache.K) {
		return mismatch("K", std::to_string(K), std::to_string(cache.K));
	}
	if (S != cache.S) {
		return mismatch("S", std::to_string(S), std::to_string(cache.S));
	}
	if (seed != cache.seed) {
		return mismatch("seed", std::to_string(seed),
		                std::to_string(cache.seed));
	}
	if (std::string(vocabHash, 16) != cache.vocabHash) {
		return mismatch("vocabulary fingerprint", std::string(vocabHash, 16),
		                cache.vocabHash);
	}
	if (std::string(splitHash, 16) != cache.splitHash) {
		return mismatch("split.txt hash", std::string(splitHash, 16),
		                cache.splitHash);
	}
	std::vector<float> row(K);
	// Ανάγνωση των εγγραφών της cache
	for (uint64_t i = 0; i < count; ++i) {
		BovwCache::Entry e;
		uint8_t valid = 0;
		// Ανάγνωση του μεγέθους της εικόνας και της ένδειξης ύπαρξης BoVW
		if (!get(in, e.fileSize) || !get(in, valid) || valid > 1) {
			error = bad + "is corrupted (truncated or invalid entry " +
			        std::to_string(i) + ")";
			return false;
		}
		// Ανάγνωση του BoVW διανύσματος αν υπάρχει
		if (valid) {
			if (!in.read(reinterpret_cast<char*>(row.data()),
			             sizeof(float) * K)) {
				error = bad + "is corrupted (truncated vector " +
				        std::to_string(i) + ")";
				return false;
			}
			// Έλεγχος για NaN ή άπειρες τιμές στο BoVW διάνυσμα
			for (float value : row) {
				if (!std::isfinite(value)) {
					error = bad +
					        "is corrupted (NaN or infinite values in entry " +
					        std::to_string(i) + ")";
					return false;
				}
			}
			// Αποθήκευση της θέσης του νέου BoVW διανύσματος
			e.row = cache.vectors.rows;
			// Προσθήκη του διανύσματος στον πίνακα της cache
			cache.vectors.data.insert(cache.vectors.data.end(), row.begin(),
			                          row.end());
			++cache.vectors.rows;
		}
		// Προσθήκη της εγγραφής στη λίστα της cache
		cache.entries.push_back(e);
	}
	// Έλεγχος αν υπάρχουν επιπλέον δεδομένα στο αρχείο
	if (in.peek() != std::ifstream::traits_type::eof()) {
		error = bad + "is corrupted (unexpected data after the last entry)";
		return false;
	}
	return true;
}

// Αρχικοποίηση των πινάκων αναζήτησης αν είναι άδειοι
void initIfNew(SearchData& data, int K) {
	if (data.database.rows == 0) {
		data.database = Matrix(0, K);
	}
	if (data.queries.rows == 0) {
		data.queries = Matrix(0, K);
	}
}

// Υπολογισμός BoVW για μία εικόνα
bool imageBovw(const std::string& path, const Matrix& vocabulary, Matrix& bovw,
               bool& empty, std::string& error) {
	cv::Mat gray;
	std::vector<cv::KeyPoint> kps;
	std::vector<float> descriptors;
	// Εξαγωγή των SIFT descriptors της εικόνας
	if (!extractSiftDescriptors(path, gray, kps, descriptors, error)) {
		return false;
	}
	// Έλεγχος αν η εικόνα δεν έχει SIFT descriptors
	empty = descriptors.empty();
	if (!empty) {
		// Μετατροπή των descriptors σε Matrix
		Matrix desc(static_cast<int>(descriptors.size() / 128), 128);
		desc.data = std::move(descriptors);
		// Υπολογισμός του BoVW histogram
		bovw = computeBovw(desc, vocabulary);
	}
	return true;
}

bool cachedBovw(BovwCache* cache, size_t index, const std::string& path,
                const Matrix& vocabulary, SearchData& data, Matrix& bovw,
                bool& empty, std::string& error) {
	if (!cache) {
		++data.imagesComputed;
		return imageBovw(path, vocabulary, bovw, empty, error);
	}
	std::error_code ec;
	const uint64_t size = fs::file_size(path, ec);
	if (ec) {
		error = "cannot read image '" + path + "' (missing or unreadable)";
		return false;
	}
	if (index < cache->entries.size()) {
		const BovwCache::Entry& e = cache->entries[index];
		if (e.fileSize != size) {
			error = "BoVW cache '" + cache->path +
			        "' does not match the images: '" + path + "' has " +
			        std::to_string(size) + " bytes but the cache saw " +
			        std::to_string(e.fileSize) + " (another dataset?)";
			return false;
		}
		empty = e.row < 0;
		if (!empty) {
			// Αντιγραφή του αποθηκευμένου BoVW διανύσματος από την cache
			bovw = Matrix(1, cache->vectors.cols);

			for (int j = 0; j < cache->vectors.cols; ++j) {
				bovw.row(0)[j] = cache->vectors.row(e.row)[j];
			}
		}
		++data.imagesFromCache;
		return true;
	}
	if (index != cache->entries.size()) {
		error = "internal error: BoVW cache entries must be added in order";
		return false;
	}
	if (!imageBovw(path, vocabulary, bovw, empty, error)) return false;
	BovwCache::Entry e;
	e.fileSize = size;
	if (!empty) {
		e.row = cache->vectors.rows;
		// Προσθήκη του νέου BoVW διανύσματος στην cache
		cache->vectors.data.insert(cache->vectors.data.end(), bovw.data.begin(),
		                           bovw.data.end());
		++cache->vectors.rows;
	}
	cache->entries.push_back(e);
	++cache->unsaved;
	++data.imagesComputed;
	return true;
}

bool addHpatches(const std::string& hpPath, const std::string& splitPath,
                 const std::string& set, const Matrix& vocabulary,
                 SearchData& data, std::string& error, BovwCache* cache) {
	if (set != "validation" && set != "test") {
		error = "set must be validation or test";
		return false;
	}
	// Ανάγνωση των sequences που ανήκουν στο επιλεγμένο σύνολο
	std::vector<std::string> sequences;
	if (!readSplitSequences(splitPath, set, sequences, error)) return false;
	if (sequences.empty()) {
		error = "no '" + set + "' sequences in '" + splitPath + "'";
		return false;
	}
	// Αρχικοποίηση των πινάκων με βάση το μέγεθος του vocabulary
	initIfNew(data, vocabulary.rows);
	size_t index =
	    0;  // position in the cache: sequences in split order, images 1..6
	// Επεξεργασία κάθε sequence
	for (const std::string& seq : sequences) {
		std::vector<int> rows;
		// Κάθε sequence περιέχει 6 εικόνες
		for (int k = 1; k <= 6; ++k) {
			const std::string id = seq + "/" + std::to_string(k);
			Matrix bovw;
			bool empty = false;
			// Ανάκτηση του BoVW από την cache ή υπολογισμός του
			if (!cachedBovw(
			        cache, index++,
			        hpPath + "/" + seq + "/" + std::to_string(k) + ".ppm",
			        vocabulary, data, bovw, empty, error)) {
				return false;
			}
			// Απόρριψη εικόνας χωρίς SIFT descriptors
			if (empty) {
				error = "HPatches image " + id +
				        " has no SIFT descriptors (no replacement rule exists "
				        "for HPatches)";
				return false;
			}
			if (k == 1) {
				data.queryIds.push_back(seq);
				// Προσθήκη του BoVW διανύσματος στα queries
				data.queries.data.insert(data.queries.data.end(),
				                         bovw.data.begin(), bovw.data.end());
				++data.queries.rows;
			} else {
				rows.push_back(data.database.rows);
				data.dbIds.push_back(id);
				// Προσθήκη του BoVW διανύσματος στη βάση εικόνων
				data.database.data.insert(data.database.data.end(),
				                          bovw.data.begin(), bovw.data.end());
				++data.database.rows;
				++data.hpatchesImages;
			}
		}
		data.relevant.push_back(rows);
	}
	return true;
}

// Προσθήκη D έγκυρων εικόνων MIRFlickr στη βάση δεδομένων
bool addDistractors(const std::string& mirPath, int D, int total,
                    const Matrix& vocabulary, SearchData& data,
                    std::string& error, BovwCache* cache) {
	// Αρχικοποίηση της βάσης με βάση το μέγεθος του vocabulary
	initIfNew(data, vocabulary.rows);
	int taken = 0;
	// Επεξεργασία εικόνων μέχρι να συγκεντρωθούν D έγκυρες
	for (int i = 1; i <= total && taken < D; ++i) {
		const std::string name = "im" + std::to_string(i) + ".jpg";
		Matrix bovw;
		bool empty = false;
		// Ανάκτηση του BoVW από την cache ή υπολογισμός του
		if (!cachedBovw(cache, i - 1, mirPath + "/" + name, vocabulary, data,
		                bovw, empty, error)) {
			return false;
		}
		// Περιοδική αποθήκευση της cache κάθε 2000 νέες εικόνες
		if (cache && cache->unsaved >= 2000 && !saveBovwCache(*cache, error)) {
			return false;  // checkpoint
		}
		// Παράλειψη εικόνων χωρίς SIFT descriptors
		if (empty) {
			++data.skippedDistractors;
			continue;
		}
		// Προσθήκη της έγκυρης εικόνας στη βάση δεδομένων
		data.dbIds.push_back(name);
		data.database.data.insert(data.database.data.end(), bovw.data.begin(),
		                          bovw.data.end());
		++data.database.rows;
		++taken;
	}
	// Έλεγχος ότι συγκεντρώθηκαν D έγκυρες εικόνες
	if (taken < D) {
		error = "only " + std::to_string(taken) + " of the requested " +
		        std::to_string(D) +
		        " MIRFlickr images have SIFT descriptors (" +
		        std::to_string(total) + " files examined)";
		return false;
	}
	return true;
}

bool prepareSearchData(const Config& cfg, SearchData& data,
                       VocabInfo& vocabInfo, std::string& error) {
	// Αρχικοποίηση των δεδομένων αναζήτησης
	data = SearchData();
	// Φόρτωση του vocabulary από την cache ή εκπαίδευση και αποθήκευσή του
	std::vector<float> vocabValues;
	if (!trainOrLoadVocabulary(cfg.hp, cfg.split, cfg.vocab, cfg.S, cfg.seed,
	                           vocabCachePath(cfg.vocab, cfg.S, cfg.seed),
	                           vocabValues, vocabInfo, error)) {
		return false;
	}
	// Μετατροπή του vocabulary σε Matrix με K κέντρα και 128 διαστάσεις
	Matrix vocabulary(cfg.vocab, 128);
	vocabulary.data = std::move(vocabValues);
	// Δημιουργία των caches για τις εικόνες HPatches και MIRFlickr
	BovwCache hp, mir;
	// Δημιουργία κοινού τμήματος ονόματος για τα αρχεία της cache
	const std::string stem = "_K" + std::to_string(cfg.vocab) + "_S" +
	                         std::to_string(cfg.S) + "_seed" +
	                         std::to_string(cfg.seed) + ".bin";
	// Αρχικοποίηση των πληροφοριών των δύο caches
	for (BovwCache* c : {&hp, &mir}) {
		c->K = cfg.vocab;
		c->S = cfg.S;
		c->seed = cfg.seed;
		c->siftConfig = siftConfigId();
		// Υπολογισμός hash από τα δεδομένα του vocabulary
		c->vocabHash = hashBytes(
		    reinterpret_cast<const unsigned char*>(vocabulary.data.data()),
		    vocabulary.data.size() * sizeof(float));
		// Υπολογισμός hash του αρχείου split
		if (!fileHash(cfg.split, c->splitHash, error)) {
			return false;
		}
	}
	// Ορισμός των πληροφοριών της cache για το HPatches
	hp.label = "hpatches/" + cfg.set;
	hp.path = "data/bovw_hp_" + cfg.set + "_" + siftConfigId() + stem;
	// Ορισμός των πληροφοριών της cache για το MIRFlickr
	mir.label = "mirflickr";
	mir.path = "data/bovw_mir_" + siftConfigId() + stem;
	// Φόρτωση των αποθηκευμένων BoVW διανυσμάτων από τις caches
	if (!loadBovwCache(hp, error) || !loadBovwCache(mir, error)) {
		return false;
	}
	// Δημιουργία των queries και της βάσης δεδομένων
	std::string buildError, saveError;
	bool ok = addHpatches(cfg.hp, cfg.split, cfg.set, vocabulary, data,
	                      buildError, &hp) &&
	          addDistractors(cfg.mir, cfg.D, kMirflickrImages, vocabulary, data,
	                         buildError, &mir);
	// Αποθήκευση των νέων δεδομένων στις caches ακόμα και αν αποτύχει η
	// δημιουργία της βάσης
	for (BovwCache* c : {&hp, &mir}) {
		if (c->unsaved > 0 && !saveBovwCache(*c, saveError)) {
			error = saveError;
			return false;
		}
	}
	// Μεταφορά του σφάλματος αν απέτυχε η δημιουργία των δεδομένων
	if (!ok) {
		error = buildError;
	}
	return ok;
}