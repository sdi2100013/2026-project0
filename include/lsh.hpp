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

