#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "args.hpp"
#include "kmeans.hpp"  // Matrix, kmeans, nearestCentroid

std::string siftConfigId();
std::string hashBytes(const void* data, size_t size);
bool fileHash(const std::string& path, std::string& hash, std::string& error);

constexpr int kIvfTopK = 10;  // ιδιο με το kTopK

// Inverted file index πανω στη βαση n x K (Matrix): coarse centroids απο τον
// k-means μας σε ΟΛΑ τα n διανυσμτα και μια λιστα δεικτων της βασης (αυξουσα)
// ανα centroid. Το IVFPQ κρατα επιπλεον, για καθε στοιχειο της λιστας, τον
// κωδικο PQ του residual x - c(x): M δεικτες σε M codebooks με 2^nbits
// centroids διασταης dsub = K / M
struct IvfIndex {
	Matrix coarse;                        // kclusters x K
	std::vector<std::vector<int>> lists;  // lists[j]: δεικτες της βασης με
	                                      // πλησιεστερο coarse centroid το j
	int M = 0, nbits = 0, dsub = 0;       // μονο PQ (M = 0 για IVFFlat)
	std::vector<Matrix>
	    codebooks;  // PQ: M πινακες (2^nbits x dsub), το codebook i κβαντιζει
		            // τις στηλες [i*dsub, (i+1)*dsub)
	std::vector<std::vector<uint16_t>>
	    codes;  // PQ: codes[j][pos * M + i] = κωδικος του υπο-διανυσματος i του
		        // lists[j][pos]
};

// Εκπαιδευη τα coarse centroids (kmeans, seed) σε ολη τη βαση και γεμιζει τις
// λιστες. Ριχνει std::invalid_argument αν δεν ισχυη 1 <= kclusters <= n.
IvfIndex buildIvfFlat(const Matrix& database, int kclusters, unsigned seed);

// Ελεγχος παραμετρων IVFPQ πριν γινει οποιαδηποτε δεσμευση: M > 0, K % M == 0,
// 1 <= nbits <= 16, kclusters <= n και 2^nbits <= n (καθε codebook εκπαιδευηται
// με k-means σε n υπο-διανυσμτα)
bool checkIvfPqParams(int K, int n, int kclusters, int M, int nbits,
                      std::string& error);

// IVFPQ: coarse centroids οπως πανω, residuals r(x) = x - c(x) ολων των n
// διανυσματων σπασμενα σε M διαδοχικα υπο-διανυσματα, ενα codebook 2^nbits
// centroids ανα θεση (k-means με seed + 1 + i για τη θεση i) στα residual
// υπο-διανυσματα ΟΛΩΝ των n, και ο κωδικος καθε διανυσματος. Ριχνει
// std::invalid_argument για παραμετρους που αποριπτει η checkIvfPqParams.
IvfIndex buildIvfPq(const Matrix& database, int kclusters, int M, int nbits,
                    unsigned seed);

// Look-up table για ενα residual του query: lut[i * 2^nbits + h] = ||residual_i
// - codebook_i[h]||^2 (τετραγωνικες αποστασεις, αθροιζονται αργοτερα), residual
// = q - c_j ενος coarse centroid j.
std::vector<double> pqLookupTable(const IvfIndex& index, const float* residual);

// ADC: αθροισμα lut[i * 2^nbits + code[i]] για i < M, προσεγγιση του ||q -
// x||^2.
double adcDistance(const std::vector<double>& lut, const uint16_t* code, int M,
                   int nbits);

// Αναζητιση IVFPQ: `query` ειναι K floats. Για καθενα απο τα nprobe πλησιεστερα
// coarse centroids c_j δικο του residual q - c_j και δικο του LUT, ADC για καθε
// στοιχειο της λιστας j, και μετα τα καλυτερα topK κατα ADC (ισοπαλιες:
// μικροτερος δεικτης βασης). Δεν γινεται re-ranking με τα αρχικα διανυσματα.
// Ποτε padding. Ριχνει std::invalid_argument αν το index δεν εχει PQ
std::vector<int> ivfPqSearch(const IvfIndex& index, const float* query,
                             int nprobe, int topK = kIvfTopK);

// Τα min(nprobe, kclusters) coarse centroids πλησιεστερα στο q (K floats,
// squared Euclidean), πλησιεστερο πρωτο, ισοπαλιες με μικροτερο δεικτη. Κοινο
// για IVFFlat και IVFPQ.
std::vector<int> nearestCoarse(const Matrix& coarse, const float* q,
                               int nprobe);

// Αναζητηση IVFFlat: πραγματικες squared αποστασεις στα διανυσματα των nprobe
// πλησιεστερων λιστων, τα καλυτερα topK. Αν υπαρχουν λιγοτεροι απο topK
// υποψηφιοι, επιστρεφονται λιγοτεροι (ποτε padding). Με nprobe = kclusters
// εξεταζεται ολη η βαση και το αποτελεσμα ισουται με exact search. `query`: K
// floats, `database`: ο πινακας πανω στον οποιο χτιστηκε το index
std::vector<int> ivfFlatSearch(const IvfIndex& index, const Matrix& database,
                               const float* query, int nprobe,
                               int topK = kIvfTopK);

// Με τι εκπαιδευτηκε ενα αποθηκευμενο index: καθε πεδιο πρεπει να ταιριαζει στο
// φορτωμα
struct IvfMeta {
	std::string method;  // "ivfflat" ή "ivfpq"
	int K = 0, n = 0,
	    kclusters = 0;     // διάσταση BoVW, μέγεθος βάσης, coarse clusters
	int M = 0, nbits = 0;  // παράμετροι PQ (0 για ivfflat)
	int S = 0, D = 0;      // παράμετροι εκτέλεσης της βάσης
	unsigned seed = 0;
	std::string set,
	    splitHash;  // "validation"/"test" και αποτύπωμα του split.txt
	std::string
	    siftConfig;  // siftConfigId() των descriptors πίσω από τα διανύσματα
	std::string dbHash;  // αποτύπωμα των ίδιων των διανυσμάτων της βάσης
};

constexpr int kIvfFormatVersion = 4;

// Περιγραφη την εκτελεση: μεθοδος, μεγεθη και παραμετροι απο το cfg,
// αποτυπωματα split.txt και βασης
bool makeIvfMeta(const Config& cfg, const Matrix& database, int kclusters,
                 IvfMeta& meta, std::string& error);

// Δυαδικη cache (magic "IVFCACHE") των coarse centroids (και των PQ codebooks)·
// οι λιστες και οι κωδικοι ξαναχτιζονται απο τη βαση, που ειναι
// ντετερμινιστικο
bool saveIvf(const std::string& path, const IvfIndex& index,
             const IvfMeta& meta, std::string& error);
bool loadIvf(const std::string& path, const IvfMeta& expected,
             const Matrix& database, IvfIndex& index, std::string& error);

// Αρχειο cache αυτης της εκτελεσης κατω απο data/ (καταληξη .bin)
std::string ivfCachePath(const Config& cfg, int kclusters);

// Φορτωνει το index του cfg.method (ivfflat η ivfpq) απο την cache αν το αρχειο
// υπαρχει (οποιαδηποτε ασυμφωνια η φθορα ειναι σφαλμα), αλλιως το εκπαιδευει με
// τον k-means μας και το αποθηκευει
bool prepareIvf(const Config& cfg, const Matrix& database, IvfIndex& index,
                bool& fromCache, double& seconds, std::string& error);
