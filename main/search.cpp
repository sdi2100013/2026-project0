#include <iostream>
#include <fstream>
#include <cmath>

#include "args.hpp"
#include "sift_extract.hpp"
#include "vocabulary.hpp"
#include "database.hpp"
#include "bovw.hpp"
#include "exact.hpp"
#include "lsh.hpp"
#include "hypercube.hpp"
#include "ivfflat.hpp"
#include "ivfpq.hpp"
#include "evaluation.hpp"


int main(int argc, char* argv[]){
    Config cfg;
    std::string error;

    //Ανάγνωση και έλεγχος των παραμέτρων εκτέλεσης
    if (!parseArgs(std::vector<std::string>(argv + 1, argv + argc), cfg, error)) {
        std::cerr << "Arguments: FAIL - " << error << '\n';
        return 1;
    }

    //Προετοιμασία των δεδομένων και του vocabulary
    SearchData data;
    VocabInfo vocabInfo;

    if (!prepareSearchData(cfg, data, vocabInfo, error)) {
        std::cerr << "PrepareSearchData: FAIL - " << error << '\n';
        return 1;
    }

    std::cout << "Database: " << data.database.rows << " images\n";
    std::cout << "Queries: " << data.queries.rows << '\n';

    SearchMethod method;

    if (cfg.method == "exact") {

        //Ακριβής αναζήτηση σε ολόκληρη τη βάση
        method = [&](const Matrix& query) {
            return exactSearch(data.database, query);
        };

    } else if (cfg.method == "lsh") {

        //Κατασκευή του ευρετηρίου LSH
        LshIndex index = buildLsh(data.database, cfg.k, cfg.L, cfg.w, cfg.seed);

        //Αναζήτηση μέσω του ευρετηρίου LSH
        method = [&, index](const Matrix& query) {
            return lshSearch(index, data.database, query);
        };

    }else{
        std::cerr << "Supported methods: exact, lsh\n";
        return 1;
    }

    //Εκτέλεση όλων των queries και υπολογισμός μετρικών
    RunResult result = runQueries(data, method, 1);
    //Αποθήκευση των αποτελεσμάτων αναζήτησης στο αρχείο εξόδου
    writeResults(cfg.out, cfg.method, data, result, error);

    //Αποθήκευση των χρόνων εκτέλεσης σε αρχείο CSV
    writeTiming("data/timing_" + cfg.method + ".csv", data, result, error);

    //Εμφάνιση των μέσων αποτελεσμάτων
    std::cout << "Recall@5: " << result.mean.recall5 << '\n';
    std::cout << "Recall@10: " << result.mean.recall10 << '\n';
    std::cout << "mAP@10: " << result.mean.ap10 << '\n';
    std::cout << "Mean time (ms): " << result.meanMs << '\n';

    return 0;
}
