#pragma once

#include "database.hpp"

#include <functional>
#include <string>
#include <vector>

//Πλήθος σχετικών εικόνων ανά query
constexpr int kRelevantPerQuery = 5;

//Μετρικές αξιολόγησης ενός query
struct QueryMetrics {
    double recall5 = 0;
    double recall10 = 0;
    double ap10 = 0;
};


//Υπολογισμός των μετρικών αξιολόγησης ενός query
QueryMetrics evaluateQuery(const std::vector<int>& ranked, const std::vector<int>& relevant);

//Μέθοδος αναζήτησης που δέχεται ένα BoVW query διαστάσεων 1 x K
//Επιστρέφει τους δείκτες των εικόνων με τα καλύτερα αποτελέσματα από τη βάση
using SearchMethod = std::function<std::vector<int>(const Matrix& query)>;



//Αποτελέσματα αξιολόγησης όλων των queries
struct RunResult {
    std::vector<std::vector<int>> ranked;
    std::vector<QueryMetrics> metrics;
    std::vector<double> ms;
    std::vector<std::vector<double>> repMs;
    QueryMetrics mean;
    double meanMs = 0;
};


//Εκτέλεση της επιλεγμένης μεθόδου αναζήτησης για κάθε query
//Μετράται μόνο ο χρόνος εκτέλεσης της μεθόδου, από το έτοιμο BoVW διάνυσμα μέχρι τα αποτελέσματα
//Αν repeats > 1, όλα τα queries εκτελούνται ξανά με το ίδιο index και τις ίδιες caches
//Ο χρόνος κάθε query υπολογίζεται ως ο μέσος όρος των επαναλήψεων
//Τα αποτελέσματα πρέπει να είναι ίδια σε κάθε επανάληψη, διαφορετικά προκύπτει runtime_error
//Επιστρέφονται τα αποτελέσματα της πρώτης επανάληψης
RunResult runQueries(const SearchData& data, const SearchMethod& method, int repeats = 1);


//Αποθήκευση των χρόνων εκτέλεσης των queries
bool writeTiming(const std::string& path, const SearchData& data, const RunResult& run, std::string& error);


//Αποθήκευση των αποτελεσμάτων και των μετρικών αξιολόγησης
bool writeResults(const std::string& path, const std::string& title, const SearchData& data, const RunResult& run, std::string& error);
