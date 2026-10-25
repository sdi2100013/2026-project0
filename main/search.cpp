#include <iostream>

#include "args.hpp"
#include "sift_extract.hpp"
#include "vocabulary.hpp"
#include "exact.hpp"
#include "lsh.hpp"
#include "hypercube.hpp"
#include "ivfflat.hpp"
#include "ivfpq.hpp"
#include "evaluation.hpp"


int main(int argc, char* argv[]){

    //1. Parse command-line arguments
    Config cfg;
    std::string error;
    //Έλεγχοσ παραμέτρων
    if (!parseArgs(std::vector<std::string>(argv + 1, argv + argc), cfg, error)) {
        std::cerr << "error: " << error << "\n";
        return 2;
    }
    //Έλεγχος ότι επιλεχθηκε σωστά ο αλγοριθμος αναζήτησης
    if (cfg.method != "exact" && cfg.method != "lsh" && cfg.method != "hypercube" && cfg.method != "ivfflat" && cfg.method != "ivfpq") {
        std::cerr << "error: -" << cfg.method << " is not supported yet (implemented: -exact, -lsh, -hypercube, -ivfflat, -ivfpq)\n";
        return 1;
    } 
    
    std::mt19937 rng(42);

    std::vector<float> descriptors(10 * kSiftDim, 1.0f);
    std::vector<float> sample = sampleDescriptors(descriptors, 3, rng);

    std::cout << "SIFT sampling: "
              << (sample.size() == 3 * kSiftDim ? "PASS" : "FAIL") << '\n';

    std::cout << "SIFT config: " << siftConfigId() << '\n';

    std::vector<float> empty;
    std::cout << "Empty descriptors: "
              << (sampleDescriptors(empty, 3, rng).empty() ? "PASS" : "FAIL") << '\n';

    return 0;

    //4. Execute queries

    //5. Evaluate results

    //6. wrhite output
    return 0;
}
