#pragma once

#include <vector>

#include "kmeans.hpp"

constexpr int kTopK = 10;

// Ταξινομεί τους υποψήφιους δείκτες της database με βάση την τετραγωνική
// Ευκλείδεια απόστασή τους από το query. Οι δείκτες candidates πρέπει να είναι
// μοναδικοί Επιστρέφει τους min(candidates.size(), topK) κοντινότερους δείκτες.
// Σε περίπτωση ισοβαθμίας προηγείται ο μικρότερος δείκτης Η απόσταση κάθε
// υποψηφίου υπολογίζεται μόνο μία φορά. Η exactSearch εξετάζει όλες τις γραμμές
// της database Οι προσεγγιστικές μέθοδοι χρησιμοποιούν τη συνάρτηση για την
// κατάταξη των υποψηφίων που έχουν εντοπίσει
std::vector<int> topByDistance(const Matrix& database, const Matrix& query,
                               const std::vector<int>& candidates,
                               int topK = kTopK);

// Ακριβής αναζήτηση κοντινότερων γειτόνων
// Συγκρίνει το query (1 x K) με κάθε γραμμή της database (n x K)
// χρησιμοποιώντας την τετραγωνική Ευκλείδεια απόσταση
// Η τετραγωνική ρίζα παραλείπεται επειδή δεν αλλάζει τη σειρά κατάταξης
// Επιστρέφει τους δείκτες των min(n, topK) κοντινότερων διανυσμάτων
// Σε περίπτωση ίσης απόστασης προηγείται ο μικρότερος δείκτης
// Ρίχνει std::invalid_argument αν οι διαστάσεις δεν είναι συμβατές
std::vector<int> exactSearch(const Matrix& database, const Matrix& query,
                             int topK = kTopK);