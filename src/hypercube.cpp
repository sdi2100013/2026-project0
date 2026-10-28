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


