#include "ivf.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <utility>

namespace fs = std::filesystem;

namespace {

// ---------------------------------------------------------------------------------------------------------
// 1. Βοηθητικες συναρτησεις
// ---------------------------------------------------------------------------------------------------------

// Υπολογιζει το residual r = x - c: αφαιρει απο το διανυσμα x το centroid c,
// στοιχειο προς στοιχειο. Και τα δυο εχουν K τιμες και το αποτελεσμα γραφεται
// στο out (που πρεπει να εχει χωρο για K floats). Το residual ειναι αυτο που
// κβαντιζει το PQ: δεν κωδικοποιουμε το x, αλλα το ποσο απεχει απο το coarse
// centroid της λιστας του, γιατι ετσι οι τιμες ειναι μικροτερες και
// κβαντιζονται με μικροτερο σφαλμα.
void residual(const float* x, const float* c, int K, float* out) {
	for (int j = 0; j < K; ++j)
		out[j] = x[j] - c[j];  // μια αφαιρεση ανα διαστση
}

// Επιστρεφει νεο πινακα με τις διαδοχικες στηλες [from, from + width) καθε
// γραμμης του m. Χρησιμοποιειται για να χωρισουμε τα residuals σε M
// υπο-διανυσματα: το υπο-διανυσμα s ειναι οι στηλες [s * dsub, (s + 1) * dsub).
// Ο k-means θελει συνεχομενο πινακα, γι' αυτο γινεται αντιγραφη και οχι view
Matrix columns(const Matrix& m, int from, int width) {
	Matrix out(m.rows, width);
	for (int i = 0; i < m.rows; ++i)  // για καθε γραμμη αντιγραφουμε μονο το
	                                  // κοματι που μας ενδιαφερει
		std::copy(m.row(i) + from, m.row(i) + from + width, out.row(i));
	return out;
}

// Γεμιζει τις inverted lists (και, για IVFPQ, τους κωδικους των residuals) απο
// τα coarse centroids και τα codebooks που υπαρχουν ηδη στο index. Ειναι
// ντετερμινιστικη, γι' αυτο χρησιμοποιειται και για να ξαναχτιστει ενα index
// που φορτωθηκε απο cache: στο αρχειο αποθηκευονται μονο τα centroids και τα
// codebooks
void fillLists(IvfIndex& index, const Matrix& database) {
	const int K = database.cols;
	index.lists.assign(index.coarse.rows,
	                   {});  // μια κενη λιστα ανα coarse centroid
	// Για IVFPQ χρειαζεται και μια λιστα κωδικων ανα centroid· για IVFFlat δεν
	// υπαρχουν κωδικοι.
	index.codes.assign(index.M > 0 ? index.coarse.rows : 0, {});
	std::vector<float> r(K);  // προσωρινο residual του τρεχοντος διανυσματος
	for (int i = 0; i < database.rows; ++i) {
		const float* x = database.row(i);
		// Το διανυσμα i ανηκει στη λιστα του πλησιεστερου coarse centroid
		// (ισοπαλια: μικροτερος δεικτης).
		const int j = nearestCentroid(x, index.coarse);
		index.lists[j].push_back(i);  // οι δεικτες μπαινουν με αυξουσα σειρα
		                              // γιατι διατρεχουμε τη βαση με τη σειρα
		if (index.M == 0)
			continue;  // IVFFlat: δεν χρειαζεται κωδικος, τελειωσαμε με αυτο το
			           // διανυσμα
		// IVFPQ: κωδικοποιουμε το residual του x ως προς το δικο του coarse
		// centroid c_j.
		residual(x, index.coarse.row(j), K, r.data());
		for (int s = 0; s < index.M; ++s)
			// Το υπο-διανυσμα s του residual ξεκινα στη στηλη s * dsub. Ο
			// κωδικος ειναι ο δεικτης του πλησιεστερου στοιχειου του codebook
			// s. Το push_back κρατα τους κωδικους του διανυσματος διαδοχικα, M
			// ανα διανυσμα, με την ιδια σειρα με τη λιστα: codes[j][pos * M +
			// s].
			index.codes[j].push_back(static_cast<uint16_t>(nearestCentroid(
			    r.data() + s * index.dsub, index.codebooks[s])));
	}
}

// Απο τους υποψηφιους (δεικτες γραμμων της βασης) επιστρεφει τους topK
// πλησιεστερους στο query, με πραγματικη squared Euclidean αποσταση,
// πλησιεστερο πρωτο. Σε ισοπαλια αποστασης προηγειται ο μικροτερος δεικτης,
// ωστε το αποτελεσμα να ειναι παντα ντετερμινιστικο (και ιδιο με το exact
// search οταν εξεταζονται ολοι).
std::vector<int> rankCandidates(const Matrix& database, const float* query,
                                const std::vector<int>& candidates, int topK) {
	std::vector<std::pair<double, int>>
	    ranked;  // (squared distance, δεικτης βασης)
	ranked.reserve(candidates.size());
	for (int idx : candidates) {
		const float* x = database.row(idx);
		double s = 0;  // αθροισμα σε double ωστε να μην χανεται ακριβεια σε
		               // μεγαλες διαστασεις
		for (int j = 0; j < database.cols; ++j) {
			const double t = static_cast<double>(query[j]) - x[j];
			s += t * t;  // τετραγωνο της διαφορας στη διασταση j
		}
		ranked.push_back({s, idx});
	}
	// Αν οι υποψηφιοι ειναι λιγοτεροι απο topK επιστρεφουμε λιγοτερους (ποτε
	// padding), και αν το topK ειναι αρνητικο το κοβουμε στο 0.
	const size_t k =
	    std::min(ranked.size(), static_cast<size_t>(std::max(topK, 0)));
	// Ταξινομουμε μονο τα πρωτα k· τα ζευγαρια συγκρινονται πρωτα με την
	// αποσταση και μετα με τον δεικτη.
	std::partial_sort(ranked.begin(), ranked.begin() + k, ranked.end());
	std::vector<int> top(k);
	for (size_t i = 0; i < k; ++i)
		top[i] = ranked[i].second;  // κραταμε μονο τους δεικτες
	return top;
}

// ---------------------------------------------------------------------------------------------------------
// Μορφη του αρχειου cache (δυαδικη, με τη σειρα που γραφονται τα πεδια):
//   8 bytes   "IVFCACHE" (magic, για να αναγνωριζεται οτι ειναι αρχειο αυτου
//   του προγραμματος) int32     εκδοση της μορφης (kIvfFormatVersion) ταυτοτητα
//   method, K, n, kclusters, M, nbits, S, D, seed, set, splitHash, dbHash,
//   siftConfig
//             (τα strings ως uint32 μηκος + bytes, οι ακεραιοι ως int32, το
//             seed ως uint32)
//   coarse    int32 rows, int32 cols, μετα rows * cols floats
//   codebooks μονο για PQ: M πινακες με την ιδια διαταξη (int32 rows, int32
//   cols, floats)
// Οι λιστες και οι κωδικοι ΔΕΝ αποθηκευονται: ξαναχτιζονται απο τη βαση με τη
// fillLists
// ---------------------------------------------------------------------------------------------------------

const char kMagic[8] = {'I', 'V', 'F', 'C', 'A', 'C', 'H', 'E'};

// Γραφουν ενα signed ακεραιο 32 bit, εναν unsigned 32 bit και ενα string (μηκος
// + περιεχομενο) στο stream. Τα bytes γραφονται οπως ειναι στη μνημη, αρα το
// αρχειο διαβαζεται σωστα μονο σε μηχανημα ιδιου endianness
void putInt(std::ostream& out, int32_t v) {
	out.write(reinterpret_cast<const char*>(&v), sizeof v);
}
void putUInt(std::ostream& out, uint32_t v) {
	out.write(reinterpret_cast<const char*>(&v), sizeof v);
}
void putString(std::ostream& out, const std::string& s) {
	putUInt(out, static_cast<uint32_t>(
	                 s.size()));  // πρωτα το μηκος, για να ξερει ο αναγνωστης
					              // ποσα bytes να διαβασει
	out.write(s.data(),
	          static_cast<std::streamsize>(s.size()));  // μετα οι χαρακτηρες
}
// Γραφει εναν πινακα: διαστασεις (rows, cols) και μετα ολα τα floats
// γραμμη-γραμμη, οπως ειναι στη μνημη.
void putMatrix(std::ostream& out, const Matrix& m) {
	putInt(out, m.rows);
	putInt(out, m.cols);
	out.write(reinterpret_cast<const char*>(m.data.data()),
	          static_cast<std::streamsize>(m.data.size() * sizeof(float)));
}

// Αναγνωστης αρχειου που ΔΕΝ εμπιστευεται το περιεχομενο: θυμαται ποσα bytes
// απομενουν και καθε αναγνωση ελεγχεται απεναντι σε αυτα πριν γινει. Ετσι ενα
// κομενο η αλλοιωμενο αρχειο δεν προκαλει τεραστια δεσμευση μνημης ουτε
// αναγνωση εκτος οριων, αλλα αποτυγχανει καθαρα με μηνυμα σφαλματος.
struct Reader {
	std::ifstream in;
	size_t remaining = 0;  // bytes που δεν εχουν διαβαστει ακομα

	explicit Reader(const std::string& path) : in(path, std::ios::binary) {
		if (!in) return;  // αν δεν ανοιξε, ο καλων το ελεγχει με το `!r.in`
		in.seekg(0, std::ios::end);  // παμε στο τελος για να μαθουμε το μεγεθος
		remaining = static_cast<size_t>(in.tellg());
		in.seekg(0);  // και επιστρεφουμε στην αρχη
	}
	// Διαβαζει ακριβως n bytes στο dst. Αποτυγχανει (false) αν το αρχειο εχει
	// λιγοτερα bytes.
	bool bytes(void* dst, size_t n) {
		if (n > remaining) return false;
		in.read(static_cast<char*>(dst), static_cast<std::streamsize>(n));
		if (!in) return false;
		remaining -= n;
		return true;
	}
	bool getInt(int32_t& v) { return bytes(&v, sizeof v); }
	bool getUInt(uint32_t& v) { return bytes(&v, sizeof v); }
	// Διαβαζει string που γραφτηκε απο την putString. Το μηκος ελεγχεται πριν
	// το resize: ενα φθαρμενο μηκος (π.χ. 4 δισεκατομμυρια) θα εκανε αλλιως μια
	// τεραστια δεσμευση
	bool getString(std::string& s) {
		uint32_t n = 0;
		if (!getUInt(n) || n > remaining) return false;
		s.resize(n);
		return n == 0 || bytes(&s[0], n);
	}
};

// Αποτελεσμα της αναγνωσης ενος πινακα: επιτυχια η ο λογος αποτυχιας (ωστε ο
// καλων να δωσει σωστο μηνυμα).
enum class MatrixRead { Ok, Truncated, WrongSize, NotFinite };

// Διαβαζει εναν πινακα που περιμενουμε να εχει ακριβως rows x cols. Οι
// διαστασεις που ειναι γραμμενες στο αρχειο πρεπει να συμφωνουν με τις
// αναμενομενες, και ελεγχονται ΠΡΙΝ γινει οποιαδηποτε δεσμευση μνημης
MatrixRead getMatrix(Reader& r, int rows, int cols, Matrix& m) {
	int32_t rr = 0, cc = 0;
	if (!r.getInt(rr) || !r.getInt(cc))
		return MatrixRead::Truncated;  // δεν υπαρχουν καν οι διαστασεις
	if (rr != rows || cc != cols)
		return MatrixRead::WrongSize;  // αλλο σχημα απο το αναμενομενο
	const size_t count = static_cast<size_t>(rows) * cols;
	// Αν τα floats που θελουμε δεν χωρουν στα bytes που απομενουν, το αρχειο
	// ειναι κομενο.
	if (count > r.remaining / sizeof(float)) return MatrixRead::Truncated;
	m = Matrix(rows, cols);
	if (count > 0 && !r.bytes(m.data.data(), count * sizeof(float)))
		return MatrixRead::Truncated;
	for (float v : m.data)  // ενα centroid με NaN η απειρο θα χαλουσε σιωπηλα
	                        // ολες τις αποστασεις
		if (!std::isfinite(v)) return MatrixRead::NotFinite;
	return MatrixRead::Ok;
}

}  // namespace

// ---------------------------------------------------------------------------------------------------------
// 2. Χτισιμο των ευρετηριων
// ---------------------------------------------------------------------------------------------------------

IvfIndex buildIvfFlat(const Matrix& database, int kclusters, unsigned seed) {
	IvfIndex index;
	// Τα coarse centroids προκυπτουν απο τον k-means μας πανω σε ΟΛΑ τα n
	// διανυσματα της βασης. Αν δεν ισχυει 1 <= kclusters <= n (η τα δεδομενα
	// ειναι ακυρα), ο kmeans ριχνει std::invalid_argument και το σφαλμα φτανει
	// αυτουσιο στον καλουντα.
	index.coarse = kmeans(database, kclusters, seed).centroids;
	fillLists(
	    index,
	    database);  // καθε διανυσμα μπαινει στη λιστα του πλησιεστερου centroid
	return index;
}

bool checkIvfPqParams(int K, int n, int kclusters, int M, int nbits,
                      std::string& error) {
	// Οι ελεγχοι γινονται με τη σειρα και ο πρωτος που αποτυγχανει δινει το
	// μηνυμα. Εκτελουνται ΠΡΙΝ δεσμευτει οποιαδηποτε μνημη, ωστε μια ακυρη
	// παραμετρος να μην προκαλει τεραστιες δεσμευσεις.
	if (M <= 0 || K % M != 0)  // τα M υπο-διανυσματα πρεπει να χωριζουν ακριβως
	                           // τις K διαστασεις (dsub = K / M)
		error = "IVFPQ needs M > 0 dividing K, got K=" + std::to_string(K) +
		        ", M=" + std::to_string(M);
	else if (nbits < 1 || nbits > 16)  // οι κωδικοι αποθηκευονται σε uint16_t,
	                                   // αρα το πολυ 16 bits
		error = "IVFPQ needs 1 <= nbits <= 16, got " + std::to_string(nbits);
	else if (kclusters < 1 ||
	         kclusters > n)  // ο k-means δεν μπορει να εχει περισσοτερα
	                         // centroids απο σημεια
		error = "IVFPQ needs 1 <= kclusters <= n, got " +
		        std::to_string(kclusters) + " with n=" + std::to_string(n);
	else if ((1LL << nbits) > n)  // καθε codebook εχει 2^nbits centroids και
	                              // εκπαιδευεται σε n υπο-διανυσμτα
		error = "IVFPQ needs 2^nbits <= n to train a codebook, got 2^" +
		        std::to_string(nbits) + " > n=" + std::to_string(n);
	else
		return true;  // ολες οι παραμετροι ειναι εγκυρες
	return false;
}

IvfIndex buildIvfPq(const Matrix& database, int kclusters, int M, int nbits,
                    unsigned seed) {
	// Πρωτα οι ελεγχοι παραμετρων· αν αποτυχουν ριχνουμε εξαιρεση με το ιδιο
	// μηνυμα
	std::string error;
	if (!checkIvfPqParams(database.cols, database.rows, kclusters, M, nbits,
	                      error))
		throw std::invalid_argument(error);

	IvfIndex index;
	index.M = M;
	index.nbits = nbits;
	index.dsub = database.cols / M;  // διασταση καθε υπο-διανυσματος (ακριβης
	                                 // διαιρεση, ελεγχθηκε παραπανω)
	// Βημα 1: coarse centroids, ακριβως οπως στο IVFFlat (k-means σε ολη τη
	// βαση)
	index.coarse = kmeans(database, kclusters, seed).centroids;

	// Βημα 2: το residual x - c(x) καθε διανυσματος, οπου c(x) το πλησιεστερο
	// coarse centroid του.
	Matrix residuals(database.rows, database.cols);
	for (int i = 0; i < database.rows; ++i) {
		const float* x = database.row(i);
		residual(x, index.coarse.row(nearestCentroid(x, index.coarse)),
		         database.cols, residuals.row(i));
	}
	// Βημα 3: ενα codebook ανα θεση υπο-διανυσματος. Το codebook s εκπαιδευεται
	// με k-means (2^nbits centroids) στα υπο-διανυσματα s ΟΛΩΝ των n residuals,
	// με seed + 1 + s ωστε καθε θεση να εχει δικο της σταθερο seed (και κανενα
	// να μην ταυτιζεται με το seed των coarse centroids
	for (int s = 0; s < M; ++s)
		index.codebooks.push_back(
		    kmeans(columns(residuals, s * index.dsub, index.dsub), 1 << nbits,
			       seed + 1 + s)
		        .centroids);
	// Βημα 4: τελος, οι λιστες και οι κωδικοι ολων των διανυσματων...
	fillLists(index, database);
	return index;
}

// ---------------------------------------------------------------------------------------------------------
// 3. Αναζητιση
// ---------------------------------------------------------------------------------------------------------

std::vector<double> pqLookupTable(const IvfIndex& index,
                                  const float* residual) {
	const int s = 1 << index.nbits;  // πληθος στοιχειων σε καθε codebook
	// Ο πινακας εχει M γραμμες των s τιμων, αποθηκευμενες διαδοχικα: lut[i * s
	// + h].
	std::vector<double> lut(static_cast<size_t>(index.M) * s);
	for (int i = 0; i < index.M;
	     ++i)  // για καθε υπο-διανυσμα i του residual του query
		for (int h = 0; h < s; ++h) {  // και για καθε στοιχειο h του codebook i
			const float* c = index.codebooks[i].row(h);
			double d = 0;
			for (int j = 0; j < index.dsub; ++j) {
				// Το υπο-διανυσμα i του residual ξεκινα στη στηλη i * dsub.
				const double t =
				    static_cast<double>(residual[i * index.dsub + j]) - c[j];
				d += t * t;
			}
			lut[static_cast<size_t>(i) * s + h] =
			    d;  // τετραγονικη αποσταση υπο-διανυσματος απο το στοιχειο h
		}
	return lut;
}

double adcDistance(const std::vector<double>& lut, const uint16_t* code, int M,
                   int nbits) {
	// Asymmetric Distance Computation: η αποσταση query - διανυσμα
	// προσεγγιζεται ως αθροισμα, πανω στα M υπο-διανυσματα, της ετοιμης
	// αποστασης απο το LUT για τον κωδικο που εχει το διανυσμα σε καθε θεση. Το
	// query μενει ακριβες (asymmetric), μονο το διανυσμα της βασης ειναι
	// κβαντισμενο.
	double sum = 0;
	for (int i = 0; i < M; ++i)
		sum += lut[(static_cast<size_t>(i) << nbits) +
		           code[i]];  // i << nbits == i * 2^nbits
	return sum;
}

std::vector<int> ivfPqSearch(const IvfIndex& index, const float* query,
                             int nprobe, int topK) {
	// Το index πρεπει να ειναι πραγματικο PQ index· διαφορετικα (π.χ. IVFFlat)
	// δεν υπαρχουν codebooks και κωδικοι.
	if (index.M == 0 ||
	    index.codebooks.size() != static_cast<size_t>(index.M) ||
	    index.lists.size() != static_cast<size_t>(index.coarse.rows))
		throw std::invalid_argument("IVFPQ: needs a PQ index");
	const int K = index.coarse.cols;
	std::vector<float> r(K);  // residual του query ως προς το τρεχον centroid
	std::vector<std::pair<double, int>>
	    found;  // (αποσταση ADC, δεικτης βασης) ολων των υποψηφιων
	// Εξεταζουμε μονο τις nprobe λιστες των πλησιεστερων coarse centroids του
	// query.
	for (int j : nearestCoarse(index.coarse, query, nprobe)) {
		// Καθε λιστα j κωδικοποιηθηκε ως προς το δικο της centroid c_j, αρα και
		// το query πρεπει να μετατραπει σε residual ως προς το ΙΔΙΟ c_j. Γι'
		// αυτο υπολογιζεται ξεχωριστο residual και ξεχωριστο LUT ανα λιστα· ενα
		// κοινο LUT για ολες θα εδινε λαθος αποστασεις.
		residual(query, index.coarse.row(j), K, r.data());
		const std::vector<double> lut = pqLookupTable(index, r.data());
		for (size_t pos = 0; pos < index.lists[j].size(); ++pos)
			// Ο κωδικος της θεσης pos ξεκινα στο pos * M και εχει M στοιχεια.
			found.push_back({adcDistance(lut, &index.codes[j][pos * index.M],
			                             index.M, index.nbits),
			                 index.lists[j][pos]});
	}
	// Κραταμε τους topK με τη μικροτερη προσεγγιστικη αποσταση (ισοπαλια:
	// μικροτερος δεικτης βασης). Δεν γινεται re-ranking με τα αρχικα διανυσματα
	// και δεν γινεται padding αν οι υποψηφιοι ειναι λιγοτεροι.
	const size_t k =
	    std::min(found.size(), static_cast<size_t>(std::max(topK, 0)));
	std::partial_sort(found.begin(), found.begin() + k, found.end());
	std::vector<int> top(k);
	for (size_t i = 0; i < k; ++i) top[i] = found[i].second;
	return top;
}

std::vector<int> nearestCoarse(const Matrix& coarse, const float* q,
                               int nprobe) {
	// Υπολογιζουμε την αποσταση του q απο ΟΛΑ τα coarse centroids (ειναι λιγα:
	// kclusters ~ sqrt(n)).
	std::vector<std::pair<double, int>>
	    dist;  // (squared distance, δεικτης centroid)
	dist.reserve(coarse.rows);
	for (int j = 0; j < coarse.rows; ++j) {
		double s = 0;
		for (int c = 0; c < coarse.cols; ++c) {
			const double t = static_cast<double>(q[c]) - coarse.row(j)[c];
			s += t * t;
		}
		dist.push_back({s, j});
	}
	// Κραταμε τα nprobe πλησιεστερα· το nprobe κοβεται στο πληθος των
	// centroids, και nprobe <= 0 δινει κενη λιστα.
	const size_t k =
	    std::min(dist.size(), static_cast<size_t>(std::max(nprobe, 0)));
	std::partial_sort(
	    dist.begin(), dist.begin() + k,
	    dist.end());  // ζευγαρια: αποσταση, μετα δεικτης (ισοπαλιες)
	std::vector<int> nearest(k);
	for (size_t i = 0; i < k; ++i) nearest[i] = dist[i].second;
	return nearest;
}

std::vector<int> ivfFlatSearch(const IvfIndex& index, const Matrix& database,
                               const float* query, int nprobe, int topK) {
	if (database.cols != index.coarse.cols)
		throw std::invalid_argument(
		    "IVFFlat: database and index have different dimensions");
	// Συγκεντρωνουμε ως υποψηφιους ολα τα διανυσματα των nprobe πλησιεστερων
	// λιστων.
	std::vector<int> candidates;  // οι λιστες ειναι ξενες μεταξυ τους, αρα καθε
	                              // δεικτης εμφανιζεται μια φορα
	for (int j : nearestCoarse(index.coarse, query, nprobe))
		candidates.insert(candidates.end(), index.lists[j].begin(),
		                  index.lists[j].end());
	// Και τους ταξινομουμε με την πραγματικη αποσταση. Με nprobe = kclusters οι
	// υποψηφιοι ειναι ολη η βαση, οποτε το αποτελεσμα ισουται με το exact
	// search.
	return rankCandidates(database, query, candidates, topK);
}

// ---------------------------------------------------------------------------------------------------------
// 4. Cache
// ---------------------------------------------------------------------------------------------------------

bool makeIvfMeta(const Config& cfg, const Matrix& database, int kclusters,
                 IvfMeta& meta, std::string& error) {
	// Συμπληρωνει την ταυτοτητα της εκτελεσης: τι μεθοδος, σε τι δεδομενα και
	// με ποιες παραμετρους θα εκπαιδευτει το index. Μια cache επιτρεπεται να
	// φορτωθει μονο αν ΟΛΑ αυτα ταιριαζουν
	meta = IvfMeta();
	meta.method = cfg.method;
	meta.K = database.cols;
	meta.n = database.rows;
	meta.kclusters = kclusters;
	meta.M = cfg.method == "ivfpq"
	             ? cfg.M
	             : 0;  // οι παραμετροι PQ εχουν νοημα μονο για ivfpq
	meta.nbits = cfg.method == "ivfpq" ? cfg.nbits : 0;
	meta.S = cfg.S;
	meta.D = cfg.D;
	meta.seed = cfg.seed;
	meta.set = cfg.set;
	meta.siftConfig =
	    siftConfigId();  // οι descriptors που εδωσαν τα διανυσματα της βασης
	// Αποτυπωμα των ιδιων των τιμων της βασης: αν αλλαξει εστω ενα float, η
	// cache αποριπτεται
	meta.dbHash =
	    hashBytes(database.data.data(), database.data.size() * sizeof(float));
	// Αποτυπωμα του split.txt· αποτυγχανει (και επιστρεφει false με μηνυμα) αν
	// το αρχειο δεν διαβαζεται
	return fileHash(cfg.split, meta.splitHash, error);
}

bool saveIvf(const std::string& path, const IvfIndex& index,
             const IvfMeta& meta, std::string& error) {
	// Ελεγχος συνεπειας: το index που αποθηκευουμε πρεπει να ταιριαζει με την
	// ταυτοτητα που γραφουμε.
	if (index.coarse.rows != meta.kclusters || index.coarse.cols != meta.K ||
	    (index.M > 0 &&
	     index.codebooks.size() != static_cast<size_t>(index.M))) {
		error = "index does not match the metadata";
		return false;
	}
	// Γραφουμε πρωτα σε προσωρινο αρχειο και στο τελος το μετονομαζουμε. Ετσι,
	// αν το προγραμμα διακοπει στη μεση, δεν μενει μισογραμμενη cache με το
	// τελικο ονομα (που την επομενη φορα θα φαινοταν φθαρμενη)..
	const std::string tmp = path + ".tmp";
	{
		std::ofstream out(tmp, std::ios::binary | std::ios::trunc);
		if (!out) {
			error = "cannot write '" + tmp + "'";
			return false;
		}
		out.write(kMagic, sizeof kMagic);  // 1. magic
		putInt(out, kIvfFormatVersion);    // 2. εκδοση μορφης
		putString(out, meta.method);  // 3. ταυτοτητα, με την ιδια σειρα που
		                              // διαβαζεται στη loadIvf
		putInt(out, meta.K);
		putInt(out, meta.n);
		putInt(out, meta.kclusters);
		putInt(out, meta.M);
		putInt(out, meta.nbits);
		putInt(out, meta.S);
		putInt(out, meta.D);
		putUInt(out, meta.seed);
		putString(out, meta.set);
		putString(out, meta.splitHash);
		putString(out, meta.dbHash);
		putString(out, meta.siftConfig);
		putMatrix(out, index.coarse);  // 4. coarse centroids
		if (index.M > 0)               // 5. τα M codebooks, μονο για PQ
			for (const Matrix& book : index.codebooks) putMatrix(out, book);
		out.flush();
		if (!out) {  // π.χ. γεματος δισκος: κανενα απο τα write παραπανω δεν
		             // επεστρεφε σφαλμα μονο του
			error = "cannot write '" + tmp + "'";
			return false;
		}
	}  // εδω κλεινει το αρχειο, πριν τη μετονομασια
	std::error_code ec;
	fs::rename(
	    tmp, path,
	    ec);  // ατομικη αντικατασταση: η υπαρχει ολκληρη η νεα cache η τιποτα
	if (ec) {
		error = "cannot move '" + tmp + "' to '" + path + "': " + ec.message();
		return false;
	}
	return true;
}

bool loadIvf(const std::string& path, const IvfMeta& want,
             const Matrix& database, IvfIndex& index, std::string& error) {
	const std::string bad = "index cache '" + path +
	                        "' ";  // κοινο προθεμα ολων των μηνυματων σφαλματος
	std::error_code ec;
	if (!fs::exists(path, ec)) {
		error = bad + "does not exist";
		return false;
	}
	Reader r(path);
	if (!r.in) {
		error = bad + "cannot be opened or parsed (corrupted?)";
		return false;
	}
	// Μικρος βοηθος για τα σφαλματα φθορας: γραφει το μηνυμα και επιστρεφει
	// false.
	auto corrupted = [&](const std::string& why) {
		error = bad + "is corrupted: " + why;
		return false;
	};

	// --- Βημα 1: magic και εκδοση. Ενα αρχειο που δεν ξεκινα με "IVFCACHE" δεν
	// ειναι cache αυτου του προγραμματος
	char magic[sizeof kMagic];
	if (!r.bytes(magic, sizeof magic) ||
	    std::memcmp(magic, kMagic, sizeof magic) != 0)
		return corrupted("not an IVF cache file");
	int32_t version = 0;
	if (!r.getInt(version)) return corrupted("missing field 'version'");
	if (version != kIvfFormatVersion) {  // αλλη εκδοση μορφης: ειναι ασυμβατη,
	                                     // οχι φθαρμενη
		error = bad + "is incompatible: format version is " +
		        std::to_string(version) + " in the file but " +
		        std::to_string(kIvfFormatVersion) + " is required";
		return false;
	}

	// --- Βημα 2: διαβαζουμε ολοκληρη την ταυτοτητα που ειναι γραμμενη στο
	// αρχειο.
	IvfMeta got;
	int32_t K = 0, n = 0, kc = 0, M = 0, nbits = 0, S = 0, D = 0;
	uint32_t seed = 0;
	if (!r.getString(got.method) || !r.getInt(K) || !r.getInt(n) ||
	    !r.getInt(kc) || !r.getInt(M) || !r.getInt(nbits) || !r.getInt(S) ||
	    !r.getInt(D) || !r.getUInt(seed) || !r.getString(got.set) ||
	    !r.getString(got.splitHash) || !r.getString(got.dbHash) ||
	    !r.getString(got.siftConfig))
		return corrupted("truncated header");  // το αρχειο τελειωνει πριν απο
		                                       // το τελος της ταυτοτητας
	got.K = K, got.n = n, got.kclusters = kc, got.M = M, got.nbits = nbits,
	got.S = S, got.D = D, got.seed = seed;

	// --- Βημα 3: συγκριση με αυτο που ζητηθηκε. Καθε διαφορα σημαινει οτι η
	// cache ειναι απο αλλη εκτελεση και δεν πρεπει να χρησιμοποιηθει· το μηνυμα
	// λεει ποιο πεδιο διαφερει και με ποιες τιμες
	auto mismatch = [&](const std::string& what, const std::string& have,
	                    const std::string& wanted) {
		error = bad + "is incompatible: " + what + " is " + have +
		        " in the file but " + wanted + " was requested";
		return false;
	};
	if (got.method != want.method)
		return mismatch("method", got.method, want.method);
	if (got.K != want.K)
		return mismatch("K", std::to_string(got.K), std::to_string(want.K));
	if (got.n != want.n)
		return mismatch("database size n", std::to_string(got.n),
		                std::to_string(want.n));
	if (got.kclusters != want.kclusters)
		return mismatch("kclusters", std::to_string(got.kclusters),
		                std::to_string(want.kclusters));
	if (got.M != want.M)
		return mismatch("M", std::to_string(got.M), std::to_string(want.M));
	if (got.nbits != want.nbits)
		return mismatch("nbits", std::to_string(got.nbits),
		                std::to_string(want.nbits));
	if (got.S != want.S)
		return mismatch("S", std::to_string(got.S), std::to_string(want.S));
	if (got.D != want.D)
		return mismatch("D", std::to_string(got.D), std::to_string(want.D));
	if (got.seed != want.seed)
		return mismatch("seed", std::to_string(got.seed),
		                std::to_string(want.seed));
	if (got.set != want.set) return mismatch("set", got.set, want.set);
	if (got.splitHash != want.splitHash)
		return mismatch("split.txt hash", got.splitHash, want.splitHash);
	if (got.siftConfig != want.siftConfig)
		return mismatch("SIFT configuration", got.siftConfig, want.siftConfig);
	if (got.dbHash != want.dbHash)
		return mismatch("database fingerprint", got.dbHash, want.dbHash);

	// --- Βημα 4: η ταυτοτητα ταιριαζει με το αιτημα, αρα τα μεγεθη που
	// περιμενουμε ειναι αυτα μιας εγκυρης εκτελεσης. Ο,τι δεν συμφωνει με αυτα
	// (π.χ. kclusters εκτος οριων) σημαινει φθορα του αρχειου.
	if (database.rows != got.n || database.cols != got.K || got.kclusters < 1 ||
	    got.kclusters > got.n)
		return corrupted("sizes do not match the database");
	if (got.M < 0 ||
	    (got.M > 0 && (got.K % got.M != 0 || got.nbits < 1 || got.nbits > 16)))
		return corrupted("invalid PQ parameters");

	// --- Βημα 5: τα coarse centroids. Ο πινακας πρεπει να ειναι ακριβως
	// kclusters x K και με πεπερασμενες τιμες.
	IvfIndex loaded;  // χτιζουμε σε τοπικο index και το αντιγραφουμε στο τελος,
	                  // ωστε μια αποτυχια να μην αλλαξει το index
	switch (getMatrix(r, got.kclusters, got.K, loaded.coarse)) {
		case MatrixRead::Ok:
			break;
		case MatrixRead::NotFinite:
			return corrupted("centroids contain NaN or infinite values");
		default:
			return corrupted("coarse centroids are not " +
			                 std::to_string(got.kclusters) + " x " +
			                 std::to_string(got.K));
	}
	loaded.M = got.M;
	loaded.nbits = got.nbits;
	loaded.dsub = got.M > 0 ? got.K / got.M : 0;

	// --- Βημα 6: για PQ διαβαζουμε τα M codebooks, καθενα 2^nbits x dsub.
	for (int i = 0; i < got.M; ++i) {
		Matrix book;
		switch (getMatrix(r, 1 << got.nbits, loaded.dsub, book)) {
			case MatrixRead::Ok:
				break;
			case MatrixRead::NotFinite:
				return corrupted("codebooks contain NaN or infinite values");
			default:
				return corrupted("codebooks are not " +
				                 std::to_string(got.M << got.nbits) + " x " +
				                 std::to_string(loaded.dsub));
		}
		loaded.codebooks.push_back(std::move(book));
	}
	// Μετα το τελευταιο πεδιο δεν πρεπει να περισσευουν bytes· αλλιως το αρχειο
	// εχει αλλοιωθει η ειναι αλλης μορφης
	if (r.remaining != 0) return corrupted("unexpected data after the index");

	// --- Βημα 7: ξαναχτιζουμε τις λιστες και τους κωδικους απο τη βαση και
	// παραδιδουμε το ετοιμο index.
	fillLists(loaded, database);
	index = std::move(loaded);
	return true;
}

std::string ivfCachePath(const Config& cfg, int kclusters) {
	// Το όνομα του αρχείου περιέχει τις βασικές παραμέτρους της εκτέλεσης, ώστε
	// διαφορετικές ρυθμίσεις να έχουν διαφορετικά αρχεία. Η αυστηρή
	// διαφοροποίηση γίνεται πάντως από την ταυτότητα μέσα στο αρχείο.
	std::string name = "data/ivf_" + cfg.method + "_" + siftConfigId() + "_K" +
	                   std::to_string(cfg.vocab) + "_S" +
	                   std::to_string(cfg.S) + "_D" + std::to_string(cfg.D) +
	                   "_" + cfg.set + "_seed" + std::to_string(cfg.seed) +
	                   "_kc" + std::to_string(kclusters);
	if (cfg.method == "ivfpq")
		name += "_M" + std::to_string(cfg.M) + "_b" +
		        std::to_string(cfg.nbits);  // μονο για PQ
	return name + ".bin";
}

bool prepareIvf(const Config& cfg, const Matrix& database, IvfIndex& index,
                bool& fromCache, double& seconds, std::string& error) {
	const auto start =
	    std::chrono::steady_clock::now();  // μετραμε τον χρονο ετοιμασιας
		                                   // (φορτωση η εκπαιδευση)
	const bool pq = cfg.method == "ivfpq";
	// Αν το kclusters δεν δοθηκε (0), το resolveKclusters επιλεγει αυτοματα
	// ceil(sqrt(n))
	const int kclusters = resolveKclusters(cfg, database.rows);
	// Ελεγχος παραμετρων πριν απο οποιαδηποτε εκπαιδευση η αναγνωση αρχειου:
	// για PQ ολοι οι ελεγχοι της checkIvfPqParams, για IVFFlat μονο οτι το
	// kclusters δεν ξεπερνα το μεγεθος της βασης
	if (pq ? !checkIvfPqParams(database.cols, database.rows, kclusters, cfg.M,
	                           cfg.nbits, error)
	       : kclusters > database.rows) {
		if (!pq)
			error = "kclusters (" + std::to_string(kclusters) +
			        ") exceeds the database size (" +
			        std::to_string(database.rows) + ")";
		return false;
	}
	IvfMeta
	    meta;  // ταυτοτητα της εκτελεσης (και αποτυπωματα split.txt / βασης)
	if (!makeIvfMeta(cfg, database, kclusters, meta, error)) return false;
	const std::string path = ivfCachePath(cfg, kclusters);

	std::error_code ec;
	fromCache = fs::exists(path, ec);
	if (fromCache) {
		// Υπαρχει cache: φορτωνεται. Αν δεν ταιριαζει η ειναι φθαρμενη ειναι
		// ΣΦΑΛΜΑ, δεν ξαναεκπαιδευουμε σιωπηλα, ωστε ο χρηστης να μαθει οτι
		// εχει στα χερια του ασυμβατο αρχειο
		if (!loadIvf(path, meta, database, index, error)) return false;
	} else {
		// Δεν υπαρχει cache: εκπαιδευουμε με τον k-means μας και αποθηκευουμε
		// για τις επομενες εκτελεσεις..
		index = pq ? buildIvfPq(database, kclusters, cfg.M, cfg.nbits, cfg.seed)
		           : buildIvfFlat(database, kclusters, cfg.seed);
		if (!saveIvf(path, index, meta, error)) return false;
	}
	seconds =
	    std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
	        .count();
	return true;
}