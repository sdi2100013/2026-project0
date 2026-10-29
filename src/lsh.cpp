#include "lsh.hpp"

#include <algorithm>
#include <cmath>
#include <random>
#include <stdexcept>

long long euclideanHash(const float* p, const double* v, int K, double t, double w) {
    double dot = 0;
    for (int j = 0; j < K; ++j) dot += static_cast<double>(p[j]) * v[j];
    const double h = std::floor((dot + t) / w);
    //Περιορισμός της τμής ωστε η μετρατροπή σε long long να είναι έγγυρη
    return static_cast<long long>(std::max(-9e18, std::min(9e18, h)));  // keep the cast defined
}



uint32_t combineHashes(const long long* h, const uint32_t* r, int k) {
    const long long P = kLshPrime;
    uint64_t acc = 0;
    for (int i = 0; i < k; ++i) {
        //Υπολογισμος του h mod P, ακόμακαι με h αρνητικό
        const uint64_t hm = static_cast<uint64_t>(((h[i] % P) + P) % P);
        //Συνσιασμος των hash χωρίς υπερχιλιση στον πολλαπλασιασμό
        acc = (acc + static_cast<uint64_t>(r[i]) * hm % P) % P;
    }
    return static_cast<uint32_t>(acc);
}



uint32_t lshId(const LshIndex& index, int table, const float* p) {
    //Αποθήκευση τιμών κατακερματισμου
    std::vector<long long> h(index.k);
    //Ευρεση κ συναερτήσεων κατακερματισμου για τρέχων πινακα
    for (int f = 0; f < index.k; ++f) {
        const size_t fn = static_cast<size_t>(table) * index.k + f;
        //Υπολογισμός της ευκλείδειας τιμής κατακερματισμού
        h[f] = euclideanHash(p, &index.v[fn * index.K], index.K, index.t[fn], index.w);
    }
    //Συνδυασμός τιμών σε ένα αναγνωριστικό κατακερματισμού
    return combineHashes(h.data(), &index.r[static_cast<size_t>(table) * index.k], index.k);
}




LshIndex buildLsh(const Matrix& database, int k, int L, double w, unsigned seed) {
    if (k < 1 || L < 1 || !(w > 0)){
        throw std::invalid_argument("LSH: k and L must be >= 1 and w > 0");
    }
    //Έλεγχος οτι η βάση έχει έγκυρεσ διαστάσεις
    if (database.rows <= 0 || database.cols <= 0){
        throw std::invalid_argument("LSH: database must have positive dimensions");
    }
    LshIndex index;
    index.K = database.cols;
    index.k = k;
    index.L = L;
    index.w = w;
    index.tableSize = std::max(1, database.rows / 4);
    std::mt19937 rng(seed);
    std::normal_distribution<double> normal(0.0, 1.0);
    std::uniform_real_distribution<double> offset(0.0, w);
    std::uniform_int_distribution<uint32_t> multiplier(1, kLshPrime - 1);
    const size_t functions = static_cast<size_t>(L) * k;
    index.v.resize(functions * index.K);
    index.t.resize(functions);
    index.r.resize(functions);
    for (double& x : index.v){
        x = normal(rng);
    }    
    for (size_t f = 0; f < functions; ++f) {
        index.t[f] = offset(rng);
        index.r[f] = multiplier(rng);
    }
    index.tables.assign(L, std::vector<std::vector<std::pair<uint32_t, int>>>(index.tableSize));
    for (int l = 0; l < L; ++l)
        for (int i = 0; i < database.rows; ++i) {
            //Υπολογισμός id τηςεικόνας στο συγκεκιμενο πίνακα
            const uint32_t id = lshId(index, l, database.row(i));
            index.tables[l][id % index.tableSize].push_back({id, i});
        }
    return index;
}


//Βρίσκει υποψήφιουσ γείτονες
std::vector<int> lshCandidates(const LshIndex& index, const float* q) {
    //Λίστα με τους δείκτες των υποψήφιων εικόνων
    std::vector<int> found;
    //Αναζήτηση του query σε όλους τους πίνακες κατακερματισμού
    for (int l = 0; l < index.L; ++l) {
        //Υπολογισμός του αναγνωριστικού του query στον πίνακα l
        const uint32_t id = lshId(index, l, q);
        //Εξέταση των εικόνων που βρίσκονται στο αντίστοιχο bucket
        for (const auto& entry : index.tables[l][id % index.tableSize]) {
            //Κρατάμε μόνο τις εικόνες με ίδιο πλήρες αναγνωριστικό
            if (entry.first == id) {
                found.push_back(entry.second);
            }
        }
    }
    //Ταξινόμηση των υποψήφιων δεικτών σε αύξουσα σειρά
    std::sort(found.begin(), found.end());
    //Αφαίρεση διπλότυπων δεικτών που βρέθηκαν σε πολλούς πίνακες
    found.erase(std::unique(found.begin(), found.end()), found.end());
    return found;
}


std::vector<int> lshSearch(const LshIndex& index, const Matrix& database, const Matrix& query, int topK) {
    //Ελένχει ότι το query έχει σωστές διαστάσεις
    if (query.rows != 1 || query.cols != index.K) {
        throw std::invalid_argument("LSH: query must be a 1 x K row");
    }
    //Βρίσκουμε τις υποψηφιες εικόνες μέσω πινάκων lsh
    std::vector<int> candidates = lshCandidates(index, query.row(0));
    //Ταξινομoύμε τους υποχήφιους βάση τη εκλειδιας απόστασης και επιστρέφουμε του Κ καλύτερους(πιο κοντά)
    return topByDistance(database, query, candidates, topK);
}