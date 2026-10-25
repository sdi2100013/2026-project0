#include <iostream>

#include "args.hpp"
#include "dataset.hpp"
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
    
    //2. Load dataset / create vectors

    //4. Execute queries

    //5. Evaluate results

    //6. wrhite output
    return 0;
}
