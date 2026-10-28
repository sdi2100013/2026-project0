#pragma once

#include "exact.hpp"

#include <cstdint>
#include <functional>
#include <unordered_map>
#include <vector>

//Δυαδικός υπερκύβος διάστασης d <= 30
//Κάθε διάνυσμα p αντιστοιχίζεται σε μία κορυφή με bits f_i(h_i(p))
//Η h_i είναι η Ευκλείδεια συνάρτηση κατακερματισμού του LSH
//Η f_i αντιστοιχίζει σταθερά κάθε ακέραιο σε 0 ή 1 με ψευδοτυχαίο τρόπο
//Αποθηκεύονται μόνο οι κορυφές που περιέχουν τουλάχιστον ένα διάνυσμα
struct HypercubeIndex {
    int K = 0, d = 0;
    double w = 0;

    //Διανύσματα προβολής και μετατοπίσεις των συναρτήσεων κατακερματισμού
    std::vector<double> v, t;

    //Μία ψευδοτυχαία τιμή για κάθε συνάρτηση f_i
    std::vector<uint64_t> salt;

    //Αντιστοίχιση κωδικού κορυφής στους δείκτες των διανυσμάτων της βάσης
    std::unordered_map<uint32_t, std::vector<int>> vertices;
};


//Κατασκευή του ευρετηρίου Hypercube
//Απαιτούνται 1 <= kproj <= 30, w > 0 και έγκυρες διαστάσεις βάσης
HypercubeIndex buildHypercube(const Matrix& database, int kproj, double w, unsigned seed);


//Υπολογισμός ενός bit από την αντίστοιχη συνάρτηση κατακερματισμού
int hypercubeBit(const HypercubeIndex& index, int fn, long long hashValue);

//Υπολογισμός του κωδικού κορυφής για ένα διάνυσμα
uint32_t hypercubeVertex(const HypercubeIndex& index, const float* p);



//Επίσκεψη έως probes κορυφών με αύξουσα απόσταση Hamming από τη βάση
//Για ίση απόσταση ακολουθείται λεξικογραφική σειρά των θέσεων των bits
//Η επίσκεψη σταματά νωρίτερα αν η visit επιστρέψει false
void forEachVertexByHamming(uint32_t base, int d, int probes, const std::function<bool(uint32_t)>& visit);

//Επιστροφή των κορυφών σε διάνυσμα με σειρά απόστασης Hamming
std::vector<uint32_t> verticesByHamming(uint32_t base, int d, int probes);


//Εύρεση υποψήφιων διανυσμάτων από έως probes κορυφές
//Εξετάζονται το πολύ M διανύσματα συνολικά
//Στο όριο probes υπολογίζονται και οι κενές κορυφές
std::vector<int> hypercubeCandidates(const HypercubeIndex& index, const float* q, int M, int probes);

//Αναζήτηση και ταξινόμηση υποψηφίων με βάση την πραγματική απόσταση
//Επιστρέφονται έως topK αποτελέσματα χωρίς συμπλήρωση
std::vector<int> hypercubeSearch( const HypercubeIndex& index, const Matrix& database, const Matrix& query, int M, int probes, int topK = kTopK);