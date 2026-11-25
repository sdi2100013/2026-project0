#pragma once

#include <limits>
#include <string>
#include <vector>

#include "feature_store.hpp"  // ImageFeatures, Matrix

// SIFT descriptor matching με Lowe ratio test (Q → I, δική μας υλοποίηση).
//
// Μαθηματική μέθοδος:
//  * Για κάθε descriptor d της Q βρίσκονται ΑΚΡΙΒΩΣ οι δύο πλησιέστεροι
//    descriptors της I με ευκλείδεια απόσταση (πλήρης αναζήτηση, όχι
//    προσέγγιση). Οι αποστάσεις υπολογίζονται ως άθροισμα τετραγώνων (ακριβές
//    για ακέραιους descriptors, αλλιώς σε double) και οι δύο τελικές αποστάσεις
//    d1 <= d2 ως sqrt.
//  * Ισοβαθμίες: διατάσσονται κατά (απόσταση, δείκτης I) λεξικογραφικά, άρα
//    μεταξύ ίσων αποστάσεων προηγείται ο μικρότερος δείκτης. Δύο descriptors με
//    ίδια απόσταση παράγουν d1 == d2 (συνεπώς απόρριψη).
//  * Αποδοχή ⇔ d1 < rho * d2 (αυστηρά). Ισότητα, και d1 = d2 = 0,
//  απορρίπτονται.
//  * Δεν υπάρχει cross-check ούτε περιορισμός μοναδικότητας.
//  * Αν η I έχει < 2 descriptors δεν εφαρμόζεται ratio test: καμία
//  αντιστοίχιση.
//  * Αποτελέσματα ταξινομημένα κατά αύξοντα δείκτη q. Ντετερμινιστικό, χωρίς
//  τυχαιότητα.

// Αποδεκτή αντιστοίχιση: Q.x[q], Q.y[q] ↔ I.x[i], I.y[i].
struct Match {
	int q = -1;     // δείκτης descriptor/keypoint της Q
	int i = -1;     // δείκτης descriptor/keypoint της I
	double d1 = 0;  // ευκλείδεια απόσταση πρώτου πλησιέστερου
	double d2 = 0;  // ευκλείδεια απόσταση δεύτερου πλησιέστερου
};

// Οι δύο πλησιέστεροι ενός descriptor της Q (πριν το ratio test).
struct TwoNearest {
	int i1 = -1, i2 = -1;  // δείκτες (i2 = -1 αν η I έχει < 2 descriptors)
	double d1 = std::numeric_limits<double>::infinity();
	double d2 = std::numeric_limits<double>::infinity();
};

// Οι δύο πλησιέστεροι κάθε γραμμής του dq μέσα στο di (Matrix N x 128).
// Σφάλμα για cols != 128 ή μη πεπερασμένες τιμές. Άδειο dq → κενό αποτέλεσμα.
bool twoNearest(const Matrix& dq, const Matrix& di,
                std::vector<TwoNearest>& out, std::string& error);

// Lowe ratio test πάνω στα αποτελέσματα του twoNearest, για 0 < rho < 1: οι
// αντιστοιχίσεις με d1 < rho * d2, κατά αύξοντα δείκτη q. Όταν η I έχει < 2
// descriptors (i2 = -1) δεν υπάρχει καμία.
std::vector<Match> matchesFromNearest(const std::vector<TwoNearest>& nearest,
                                      double rho);
