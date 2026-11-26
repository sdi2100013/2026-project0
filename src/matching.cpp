#include "matching.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>

namespace {

// Διασταση SIFT descriptor.
constexpr int kDim = 128;
// Αρχικη αποσταση πριν βρεθει γειτονας.
constexpr double kInf = std::numeric_limits<double>::infinity();

// GCC σε x86-64 Linux: επιλογη AVX2 ή default στην εκκινηση.
#if defined(__GNUC__) && defined(__x86_64__) && defined(__linux__)
#define SIMD_DISPATCH __attribute__((target_clones("avx2", "default")))
#else
// Κενο macro στις αλλες πλατφορμες.
#define SIMD_DISPATCH
#endif

// Τετραγωνικη ευκλειδεια αποσταση. Για bytes το αθροισμα χωραει σε int32.
// Για float, αθροιση σε double με σταθερη σειρα.
template <class T, class Sum>
inline Sum squaredDistance(const T* a, const T* b) {
	Sum total = 0;
	for (int j = 0; j < kDim; ++j) {
		// Cast πριν την αφαιρεση, για αποφυγη unsigned διαφορας.
		const Sum d = static_cast<Sum>(a[j]) - static_cast<Sum>(b[j]);
		total += d * d;
	}
	return total;
}

// Πληρης αναζητηση δυο γειτονων. Σε ισοβαθμια προηγειται ο μικροτερος δεικτης.
template <class T, class Sum>
inline __attribute__((always_inline)) void searchTwo(
    const T* dq, int nq, const T* di, int ni, std::vector<TwoNearest>& out) {
	// Ενα κενο αποτελεσμα ανα query.
	out.assign(nq, TwoNearest());
	for (int q = 0; q < nq; ++q) {
		const T* query = dq + static_cast<size_t>(q) * kDim;
		// Οι δυο μικροτερες τετραγωνικες αποστασεις.
		double best1 = kInf, best2 = kInf;
		// -1 οταν δεν εχει βρεθει γειτονας.
		int index1 = -1, index2 = -1;
		for (int i = 0; i < ni; ++i) {
			const double d = squaredDistance<T, Sum>(
			    query, di + static_cast<size_t>(i) * kDim);
			// Παραλειψη χειροτερων τιμων, διατηρηση σειρας στις ισοβαθμιες.
			if (d >= best2) continue;
			if (d < best1) {
				// Ο προηγουμενος πρωτος περναει στη δευτερη θεση.
				best2 = best1;
				index2 = index1;
				best1 = d;
				index1 = i;
			} else {
				// Νεος δευτερος, ακομη και με αποσταση ιση με τον πρωτο.
				best2 = d;
				index2 = i;
			}
		}
		// Αποθηκευση δεικτων και αποστασεων με sqrt.
		out[q] = {index1, index2, std::sqrt(best1), std::sqrt(best2)};
	}
}

// Byte descriptors με αθροισμα int32 και AVX2 dispatch.
SIMD_DISPATCH void searchBytes(const uint8_t* dq, int nq, const uint8_t* di,
                               int ni, std::vector<TwoNearest>& out) {
	searchTwo<uint8_t, int32_t>(dq, nq, di, ni, out);
}

// Float descriptors, αθροισμα σε double.
void searchFloats(const float* dq, int nq, const float* di, int ni,
                  std::vector<TwoNearest>& out) {
	searchTwo<float, double>(dq, nq, di, ni, out);
}

// Ελεγχος αν ολες οι τιμες επιτρεπουν byte αριθμητικη.
bool allBytes(const Matrix& m) {
	for (float v : m.data) {
		// Μονο ακεραιες τιμες στο [0, 255].
		if (!(v >= 0.f && v <= 255.f && v == std::floor(v))) return false;
	}
	return true;
}

}  // namespace

// Δυο κοντινοτεροι descriptors του di ανα γραμμη του dq.
bool twoNearest(const Matrix& dq, const Matrix& di,
                std::vector<TwoNearest>& out, std::string& error) {
	// Καθαρισμος παλιων αποτελεσματων πριν τους ελεγχους.
	out.clear();
	for (const Matrix* m : {&dq, &di}) {
		// Απαιτουνται 128 στηλες για μη κενο πινακα.
		if (m->rows > 0 && m->cols != kDim) {
			error = "descriptors must be N x 128";
			return false;
		}
		// Απορριψη NaN και απειρων τιμων.
		for (float v : m->data) {
			if (!std::isfinite(v)) {
				error = "non-finite descriptor value";
				return false;
			}
		}
	}
	// Κενο query, χωρις αναζητηση.
	if (dq.rows == 0) return true;
	if (allBytes(dq) && allBytes(di)) {
		// Μετατροπη σε uint8 για ακεραια αριθμητικη.
		const std::vector<uint8_t> bq(dq.data.begin(), dq.data.end());
		const std::vector<uint8_t> bi(di.data.begin(), di.data.end());
		searchBytes(bq.data(), dq.rows, bi.data(), di.rows, out);
	} else {
		// Γενικη περιπτωση: float με αθροισμα double.
		searchFloats(dq.data.data(), dq.rows, di.data.data(), di.rows, out);
	}
	return true;
}

// Lowe ratio test στα αποτελεσματα του twoNearest.
std::vector<Match> matchesFromNearest(const std::vector<TwoNearest>& nearest,
                                      double rho) {
	std::vector<Match> matches;
	for (size_t q = 0; q < nearest.size(); ++q) {
		const TwoNearest& r = nearest[q];
		// Δευτερος γειτονας και αυστηρα d1 < rho * d2.
		if (r.i2 >= 0 && r.d1 < rho * r.d2) {
			matches.push_back({static_cast<int>(q), r.i1, r.d1, r.d2});
		}
	}
	return matches;
}