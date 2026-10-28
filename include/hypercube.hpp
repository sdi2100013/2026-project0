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


//Υπολογισμός ενός bit από την αντίστοιχη συνάρτηση κατακερματισμού
int hypercubeBit(const HypercubeIndex& index, int fn, long long hashValue);

//Υπολογισμός του κωδικού κορυφής για ένα διάνυσμα
uint32_t hypercubeVertex(const HypercubeIndex& index, const float* p);

