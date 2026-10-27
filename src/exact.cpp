#include "exact.hpp"

#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <utility>

std::vector<int> topByDistance(const Matrix& database, const Matrix& query,
                               const std::vector<int>& candidates, int topK) {
	// Έλεγχος ότι το query έχει μία γραμή και ίδιες διαστάσεις με τη βάση
	if (query.rows != 1 || query.cols != database.cols) {
		throw std::invalid_argument(
		    "search: query must be a 1 x K row and the database n x K");
	}
	const int K = database.cols;
	const float* q = query.row(0);
	// Αποθήκευση της τετραγονικής απόστασης και του δείκτη κάθε υποψηφίου
	std::vector<std::pair<double, int>> ranked;
	ranked.reserve(candidates.size());
	// Υπολογισμός της απόστασης κάθε υποψηφίου από το query
	for (int idx : candidates) {
		const float* x = database.row(idx);
		double s = 0;
		// Υπολογισμός τετραγωνικής Ευκλείδιας απόστασης
		for (int j = 0; j < K; ++j) {
			const double t = static_cast<double>(q[j]) - x[j];
			s += t * t;
		}
		ranked.push_back({s, idx});
	}
	// Επιλογή των topK κοντινότερων υποψηφίων
	const size_t k =
	    std::min(ranked.size(), static_cast<size_t>(std::max(topK, 0)));
	// Ταξινόμηση πρώτα με βάση την απόσταση και έπειτα τον δείκτη
	std::partial_sort(ranked.begin(), ranked.begin() + k, ranked.end());
	// Αποθήκευση των δεικτών των καλύτερων αποτελεσμάτων
	std::vector<int> top(k);
	for (size_t i = 0; i < k; ++i) {
		top[i] = ranked[i].second;
	}
	return top;
}

std::vector<int> exactSearch(const Matrix& database, const Matrix& query,
                             int topK) {
	// Δημιουργία λίστας με όλους τους δείκτες της βάσης
	std::vector<int> all(database.rows);
	std::iota(all.begin(), all.end(), 0);
	// Ακριβής αναζήτηση εξετάζοντας όλα τα διανύσματα της βάσης
	return topByDistance(database, query, all, topK);
}