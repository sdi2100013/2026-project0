#include "hypercube.hpp"

#include "lsh.hpp"

#include <algorithm>
#include <random>
#include <stdexcept>



namespace {

//Ανάμειξη των bits ενός ακεραίου με τον αλγόριθμο SplitMix64
//Χρησιμοποιείται για την παραγωγή σταθερών ψευδοτυχαίων τιμών
uint64_t mix(uint64_t x) {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

} //Τέλος του ανώνυμου namespace


//Χρησιμοποιούμε διαφορετικό salt για κάθε συνάρτηση, ώστε να παράγονται διαφορετικές αντιστοιχίσεις.
int hypercubeBit(const HypercubeIndex& index, int fn, long long hashValue) {
    //Συνδυασμός της τιμής hash με το salt της συνάρτησης fn
    uint64_t x = index.salt[fn] ^ static_cast<uint64_t>(hashValue);
    //Ανάμειξη των bits για παραγωγή ψευδοτυχαίας τιμής
    uint64_t mixed = mix(x);
    //Επιστροφή του πιο σημαντικού bit ως 0 ή 1
    return static_cast<int>(mixed >> 63);
}


uint32_t hypercubeVertex(const HypercubeIndex& index, const float* p) {
    //Αρχικοποίηση του δυαδικού κωδικού της κορυφής
    uint32_t code = 0;
    //Υπολογισμός των d bits που καθορίζουν την κορυφή
    for (int i = 0; i < index.d; ++i) {
        //Υπολογισμός της Ευκλείδειας συνάρτησης κατακερματισμού h_i(p)
        const long long h = euclideanHash(p, &index.v[static_cast<size_t>(i) * index.K], index.K, index.t[i], index.w);
        //Μετατροπή του hash σε bit 0 ή 1 και τοποθέτησή του στη θέση i
        code |= static_cast<uint32_t>(hypercubeBit(index, i, h)) << i;
    }
    //Επιστροφή του κωδικού της κορυφής
    return code;
}



HypercubeIndex buildHypercube(const Matrix& database, int kproj, double w, unsigned seed) {
    //Έλεγχος των παραμέτρων του υπερκύβου
    if (kproj < 1 || kproj > 30 || !(w > 0)) {
        throw std::invalid_argument("hypercube: kproj must be in [1, 30] and w > 0");
    }
    //Έλεγχος ότι η βάση έχει έγκυρες διαστάσεις
    if (database.rows <= 0 || database.cols <= 0) {
        throw std::invalid_argument("hypercube: database must have positive dimensions");
    }
    //Αρχικοποίηση των παραμέτρων του ευρετηρίου
    HypercubeIndex index;
    index.K = database.cols;
    index.d = kproj;
    index.w = w;
    //Αρχικοποίηση γεννήτριας τυχαίων αριθμών
    std::mt19937 rng(seed);
    std::normal_distribution<double> normal(0.0, 1.0);
    std::uniform_real_distribution<double> offset(0.0, w);
    std::uniform_int_distribution<uint64_t> any;
    //Δημιουργία των διανυσμάτων προβολής
    index.v.resize(static_cast<size_t>(kproj) * index.K);
    for (double& x : index.v) {
        x = normal(rng);
    }
    //Δημιουργία μετατοπίσεων και ψευδοτυχαίων τιμών salt
    for (int i = 0; i < kproj; ++i) {
        index.t.push_back(offset(rng));
        index.salt.push_back(any(rng));
    }
    //Αντιστοίχιση κάθε διανύσματος της βάσης σε μία κορυφή
    for (int i = 0; i < database.rows; ++i) {
        //Υπολογισμός του κωδικού κορυφής για το διάνυσμα i
        uint32_t vertex = hypercubeVertex(index, database.row(i));
        //Αποθήκευση του δείκτη του διανύσματος στην αντίστοιχη κορυφή
        index.vertices[vertex].push_back(i);
    }
    //Επιστροφή του κατασκευασμένου ευρετηρίου
    return index;
}


void forEachVertexByHamming(uint32_t base, int d, int probes, const std::function<bool(uint32_t)>& visit) {
    //Μετρητής των κορυφών που έχουμε επισκεφθεί
    long long visited = 0;
    //Εξέταση κορυφών με αύξουσα απόσταση Hamming
    for (int h = 0; h <= d && visited < probes; ++h) {
        //Θέσεις των h bits που θα αντιστραφούν
        std::vector<int> pos(h);
        //Αρχικοποίηση της πρώτης επιλογής θέσεων
        for (int i = 0; i < h; ++i) {
            pos[i] = i;
        }
        while (visited < probes) {
            //Δημιουργία μάσκας με τα bits που θα αντιστραφούν
            uint32_t mask = 0;
            for (int p : pos) {
                mask |= 1u << p;
            }
            //Επίσκεψη της κορυφής που προκύπτει από τη μάσκα
            ++visited;
            if (!visit(base ^ mask)) {
                return;
            }
            //Εύρεση του επόμενου συνδυασμού θέσεων
            //με λεξικογραφική σειρά
            int i = h - 1;
            while (i >= 0 && pos[i] == d - h + i) {
                --i;
            }
            //Αν δεν υπάρχει άλλος συνδυασμός, προχωράμε
            //στην επόμενη απόσταση Hamming
            if (i < 0) {
                break;
            }
            ++pos[i];
            //Ενημέρωση των υπόλοιπων θέσεων του συνδυασμού
            for (int j = i + 1; j < h; ++j) {
                pos[j] = pos[j - 1] + 1;
            }
        }
    }
}



std::vector<uint32_t> verticesByHamming(uint32_t base, int d, int probes) {
    //Διάνυσμα αποθήκευσης των κορυφών που θα επισκεφθούμε
    std::vector<uint32_t> order;
    //Επίσκεψη των κορυφών με αύξουσα απόσταση Hamming
    forEachVertexByHamming(base, d, probes, [&](uint32_t v) {
        //Προσθήκη της τρέχουσας κορυφής στο αποτέλεσμα
        order.push_back(v);
        //Συνέχιση μέχρι να συμπληρωθεί το όριο probes
        return true;
    });
    //Επιστροφή των κορυφών με τη σειρά επίσκεψης
    return order;
}
