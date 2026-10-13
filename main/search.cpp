#include <iostream>

#include "config.hpp"
#include "dataset.hpp"
#include "exact.hpp"
#include "lsh.hpp"
#include "hypercube.hpp"
#include "ivfflat.hpp"
#include "ivfpq.hpp"
#include "evaluation.hpp"


int main(int argc, char* argv[]){

    //1. Parse command-line arguments
    //temp arguent test
    Config config = parseArguments(argc, argv);

    std::cout << "HPatches: " << config.hpPath << '\n';
    std::cout << "MIRFlickr: " << config.mirPath << '\n';
    std::cout << "Split: " << config.splitFile << '\n';
    std::cout << "Set: " << config.dataSet << '\n';
    std::cout << "OutputFile: " << config.outputFile << '\n';
    std::cout << "vocab: " << config.vocab << '\n';
    std::cout << "S: " << config.S << '\n';
    std::cout << "D: " << config.D << '\n';
    std::cout << "Seed: " << config.seed << '\n';
    if (config.method == Searchmethod::EXACT)
        std::cout << "Method: Exact\n";
    else if (config.method == Searchmethod::LSH){
        std::cout << "Method: LSH\n";
        std::cout << "LSH k: " << config.lsh.K << '\n';
        std::cout << "LSH L: " << config.lsh.L << '\n';
        std::cout << "LSH w: " << config.lsh.W << '\n';
    }
    else if (config.method == Searchmethod::HYPERCUBE){
        std::cout << "Method: Hypercube\n";
        std::cout << "Hypercube kproj: " << config.hypercube.kproj<< '\n';
        std::cout << "Hypercube M: " << config.hypercube.M << '\n';
        std::cout << "Hypercube w: " << config.hypercube.W << '\n';
        std::cout << "Hypercube probes: " << config.hypercube.probes << '\n';
    }
    else if (config.method == Searchmethod::IVFFLAT){
        std::cout << "Method: IVFFlat\n";
        std::cout << "IVFFlat kclusters: " << config.ivfflat.kclusters<< '\n';
        std::cout << "IVFFLat nprobe: " << config.ivfflat.nprobe << '\n';
    }
    else if (config.method == Searchmethod::IVFPQ){
        std::cout << "Method: IVFPQ\n";
        std::cout << "IVFPQ kclusters: " << config.ivfpq.kclusters<< '\n';
        std::cout << "IVFPQ nprobe: " << config.ivfpq.nprobe << '\n';
        std::cout << "IVFPQ M: " << config.ivfpq.M<< '\n';
        std::cout << "IVFPQ nbits: " << config.ivfpq.nbits << '\n';
    }
    //2. Load dataset / create vectors
    //temp dataset test
    cv::Mat descriptors = extractSIFT("test/image.jpg");
    std::cout << descriptors.rows << " x " << descriptors.cols << '\n';

    //3. Build selected search method

    //4. Execute queries

    //5. Evaluate results

    //6. wrhite output
    return 0;
}