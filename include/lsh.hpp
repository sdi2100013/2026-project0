#pragma once

#include "exact.hpp"

#include <cstdint>
#include <utility>
#include <vector>

//Ευκλείδεια συνάρτηση κατακερματισμού:
//h(p) = floor((dot(p, v) + t) / w)
//Το p περιέχει K τιμές float και το v περιέχει K τιμές double
//Χρησιμοποιείται επίσης ως βασικό στοιχείο του Hypercube
long long euclideanHash(const float* p, const double* v, int K, double t, double w);

//Συνδυασμός k τιμών κατακερματισμού σε ένα αναγνωριστικό
//Υπολογίζει το (sum r_i * h_i) mod P, όπου P = 2^32 - 5
//Οι τιμές h_i μπορεί να είναι αρνητικές
//Ο υπολογισμός γίνεται χωρίς υπερχείλιση
uint32_t combineHashes(const long long* h, const uint32_t* r, int k);

//Πρώτος αριθμός που χρησιμοποιείται στον συνδυασμό των hash
constexpr uint32_t kLshPrime = 4294967291u; //2^32 - 5

// Δομή του ευρετηρίου LSH με L πίνακες κατακερματισμού
// Κάθε πίνακας χρησιμοποιεί k ευκλείδιες συναρτήσεις κατακερματισμού
// Κάθε πίνακας έχει τις δικές του παραμέτρους v, t και r
// Οι τιμές αποθηκέυονται διαδοχικά στις θέσεις l*k + f
// Για τα διανύσματα v χρησιμοποιούνται K επιπλέων διαστάσεις
struct LshIndex {
    int K = 0;
    int k = 0;
    int L = 0;
    double w = 0;
    //Μέγεθος κάθε πίνακα, περίπου n/4
    int tableSize = 1;
    //Τα v περιέχουν L*k*K τιμές κανονικής κατανομής
    std::vector<double> v;
    //Τα t περιέχουν L*k μετατοπίσεις στο διάστημα [0, w)
    std::vector<double> t;
    //Οι πολλαπλασιαστές r βρίσκονται στο διάστημα [1, P)
    std::vector<uint32_t> r;
    //Κάθε bucket αποθηκεύει ζεύγη αναγνωριστικού και δείκτη εικόνας
    //tables[l][bucket] = (πλήρες αναγνωριστικό, δείκτης στη βάση)
    //Το bucket προκύπτει από το αναγνωριστικό modulo tableSize
    std::vector<std::vector<std::vector<std::pair<uint32_t, int>>>> tables;
};



//Υπολογισμός του σύνθετου αναγνωριστικού ενός διανύσματος στον πίνακα κατακερματισμού table
uint32_t lshId(const LshIndex& index, int table, const float* p);


//Κατασκευή του ευρετηρίου LSH για βάση δεδομένων n x K ολες οι τηχαίες τιμές παράγονται από std::mt19937(seed)
//Ρίχνει std::invalid_argument για k < 1, L < 1, w <= 0 ή μη έγκυρες διαστάσεις της βάσης
LshIndex buildLsh(const Matrix& database, int k, int L, double w, unsigned seed);


//Εύρεση μοναδικών δεικτών της βάσης που βρίσκονται στα buckets του query
//Συγκρίνεται το πλήρες αναγνοριστικό κατακερματισμού
//Οι δείκτες επιστρέφονται σε αύξουσα σειρά
std::vector<int> lshCandidates(const LshIndex& index, const float* q);

//Αναζήτηση των κοντινότερων γειτόνων μεταξύ των υποψηφίων του LSH
//Οι υποψήφιοι ταξινομούνται με βάση την πραγματική τους απόσταση
//Επιστρέφονται έως topK αποτελέσματα, ενδεχομένως λιγότερα ή κανένα
//Δεν προστίθενται τεχνητά αποτελέσματα για τη συμπλήρωση του topK
std::vector<int> lshSearch(const LshIndex& index, const Matrix& database, const Matrix& query, int topK = kTopK);