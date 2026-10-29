#include "evaluation.hpp"

#include <algorithm>
#include <stdexcept>
#include <fstream>



//Υπολογισμός των Recall@5, Recall@10 και AP@10 για ένα query
QueryMetrics evaluateQuery(const std::vector<int>& ranked, const std::vector<int>& relevant) {
    //Έλεγχος ότι κάθε query έχει ακριβώς 5 σχετικές εικόνες
    if (static_cast<int>(relevant.size()) != kRelevantPerQuery){
        throw std::invalid_argument("evaluation: a query needs exactly 5 relevant images");
    }
    QueryMetrics m;
    int hits5 = 0, hits10 = 0;
    double sum = 0;
    //Έλεγχος των πρώτων 10 αποτελεσμάτων της αναζήτησης
    for (int r = 1; r <= static_cast<int>(ranked.size()) && r <= 10; ++r) {
        if (std::find(relevant.begin(), relevant.end(), ranked[r - 1]) == relevant.end()){
            continue;
        }
        //Καταμέτρηση σχετικών εικόνων στις πρώτες 5 και 10 θέσεις
        ++hits10;
        if (r <= 5){
            ++hits5;
        }
        //Προσθήκη της precision στη θέση κάθε σχετικής εικόνας
        sum += static_cast<double>(hits10) / r;  // j / r_j
    }
    //Υπολογισμός Recall@5, Recall@10 και AP@10
    m.recall5 = static_cast<double>(hits5) / kRelevantPerQuery;
    m.recall10 = static_cast<double>(hits10) / kRelevantPerQuery;
    m.ap10 = sum / kRelevantPerQuery;
    return m;
}

//Αποθήκευση των χρόνων αναζήτησης σε αρχείο CSV
bool writeTiming(const std::string& path, const SearchData& data, const RunResult& run, std::string& error) {
    //Άνοιγμα του αρχείου για εγγραφή
    std::ofstream out(path);
    if (!out) {
        error = "cannot open timing file '" + path + "' for writing";
        return false;
    }
    //Ρύθμιση της ακρίβειας των χρόνων
    out.setf(std::ios::fixed);
    out.precision(6);
    out << "query,repetition,ms\n";
    //Εγγραφή του χρόνου κάθε query για κάθε επανάληψη
    for (size_t rep = 0; rep < run.repMs.size(); ++rep){
        for (size_t q = 0; q < data.queryIds.size(); ++q){
            out << data.queryIds[q] << "," << rep + 1 << "," << run.repMs[rep][q] << "\n";
        }
    }
    //Έλεγχος επιτυχούς εγγραφής
    out.flush();
    if (!out) {
        error = "error while writing '" + path + "'";
        return false;
    }
    return true;
}

//Αποθήκευση των αποτελεσμάτων αναζήτησης σε αρχείο
bool writeResults(const std::string& path, const std::string& title, const SearchData& data, 
                  const RunResult& run, std::string& error) {
    //Άνοιγμα του αρχείου για εγγραφή
    std::ofstream out(path);
    if (!out) {
        error = "cannot open output file '" + path + "' for writing";
        return false;
    }
    //Ρύθμιση της ακρίβειας των αριθμητικών τιμών
    out.setf(std::ios::fixed);
    out.precision(6);
    out << title << "\n";
    //Εγγραφή των αποτελεσμάτων κάθε query
    for (size_t q = 0; q < data.queryIds.size(); ++q) {
        out << "Query: " << data.queryIds[q] << "\n";
        //Εγγραφή των εικόνων με τη σειρά κατάταξης
        for (size_t r = 0; r < run.ranked[q].size(); ++r){
            out << "Rank-" << r + 1 << ": " << data.dbIds[run.ranked[q][r]] << "\n";
        }
        //Εγγραφή των μετρικών του query
        out << "Recall@5: " << run.metrics[q].recall5 << "\nRecall@10: " << run.metrics[q].recall10 
                            << "\nAP@10: " << run.metrics[q].ap10 << "\n\n";
    }
    //Εγγραφή των συνολικών μέσων τιμών
    out << "Mean Recall@5: " << run.mean.recall5 << "\nMean Recall@10: " << run.mean.recall10 
    << "\nmAP@10: " << run.mean.ap10 << "\nAverage query time: " << run.meanMs << "\n";
    //Έλεγχος επιτυχούς εγγραφής
    out.flush();
    if (!out) {
        error = "error while writing '" + path + "'";
        return false;
    }
    return true;
}



//Εκτέλεση της μεθόδου αναζήτησης για όλα τα queries
RunResult runQueries(const SearchData& data, const SearchMethod& method, int repeats) {
    using Clock = std::chrono::steady_clock;
    RunResult run;
    //Πλήθος των queries
    const int Q = data.queries.rows;
    //Αρχικοποίηση των χρόνων εκτέλεσης για κάθε επανάληψη και query
    run.repMs.assign(std::max(repeats, 1), std::vector<double>(Q, 0.0));
    //Εκτέλεση όλων των queries για κάθε επανάληψη
    for (size_t rep = 0; rep < run.repMs.size(); ++rep) {
        for (int q = 0; q < Q; ++q) {
            //Αντιγραφή του BoVW διανύσματος του query σε Matrix 1 x K
            Matrix query(1, data.queries.cols);
            for (int j = 0; j < data.queries.cols; ++j) {
                query.row(0)[j] = data.queries.row(q)[j];
            }
            //Μέτρηση μόνο του χρόνου εκτέλεσης της μεθόδου αναζήτησης
            const auto start = Clock::now();
            std::vector<int> top = method(query);
            const auto stop = Clock::now();
            //Αποθήκευση του χρόνου σε milliseconds
            run.repMs[rep][q] =
                std::chrono::duration<double, std::milli>(stop - start).count();
            if (rep == 0) {
                //Αξιολόγηση και αποθήκευση των αποτελεσμάτων της πρώτης επανάληψης
                run.metrics.push_back(evaluateQuery(top, data.relevant[q]));
                run.ranked.push_back(std::move(top));
            } else if (top != run.ranked[q]) {
                //Έλεγχος ότι οι επαναλήψεις επιστρέφουν τα ίδια αποτελέσματα
                throw std::runtime_error(
                    "the search returned different results for query " +
                    data.queryIds[q] + " in repetition " + std::to_string(rep + 1)
                );
            }
        }
    }
    //Αρχικοποίηση των μέσων χρόνων για κάθε query
    run.ms.assign(Q, 0.0);
    //Υπολογισμός των μέσων χρόνων και των συνολικών μετρικών
    for (int q = 0; q < Q; ++q) {
        for (const auto& rep : run.repMs) {
            run.ms[q] += rep[q] / run.repMs.size();
        }
        run.mean.recall5 += run.metrics[q].recall5 / Q;
        run.mean.recall10 += run.metrics[q].recall10 / Q;
        run.mean.ap10 += run.metrics[q].ap10 / Q;
        run.meanMs += run.ms[q] / Q;
    }
    return run;
}