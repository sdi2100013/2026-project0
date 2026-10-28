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
