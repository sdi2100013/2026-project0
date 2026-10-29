
#include "ivf.hpp"

#include <algorithm>
#include <stdexcept>

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