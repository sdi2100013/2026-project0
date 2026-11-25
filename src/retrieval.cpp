#include "retrieval.hpp"

#include <iostream>
#include <memory>

#include "exact.hpp"
#include "lsh.hpp"
#include "hypercube.hpp"
#include "ivf.hpp"


//προετιμασία του μηχανισμού ανάκτησης υποψήφιων ηκόνων
bool makeRetriever(const Config& cfg, const SearchData& data, TopC& retrieve, std::string& title, std::string& error) {
    //αναφορά στον πίνακα των BoVW διανυσμάτων της βάσης
    const Matrix& db = data.database;
    //περίπτωση ακριβούς αναζήτησης
    if (cfg.method == "exact") {
        title = "Exact Search";
        retrieve = [&db](const Matrix& q, int C) {
            return exactSearch(db, q, C);
        };
    } else if (cfg.method == "lsh") {
        title = "LSH";
        //κατασκευή του ευρετηρίου LSH με βάση τις επιλεγμένες παραμέτρους.
        auto index = std::make_shared<const LshIndex>(
        buildLsh(db, cfg.k, cfg.L, cfg.w, cfg.seed)
        );
        //κνάκτηση των C πρώτων υποψήφιων εικόνων μέσω LSH.
        retrieve = [index, &db](const Matrix& q, int C) {
            return lshSearch(*index, db, q, C);
        };
    } else if (cfg.method == "hypercube") {
        title = "Hypercube";
        //καατασκευή του ευρετηρίου Hypercube με βάση τις επιλεγμένες παραμέτρους.
        auto index = std::make_shared<const HypercubeIndex>(
            buildHypercube(db, cfg.kproj, cfg.w, cfg.seed)
        );
        //ανάκτηση των C πρώτων υποψήφιων εικόνων μέσω Hypercube.
        retrieve = [index, &db, M = cfg.M, probes = cfg.probes](const Matrix& q, int C) {
            return hypercubeSearch(*index, db, q, M, probes, C);
        };
    } else {
        //Προσδιορισμός της μεθόδου IVF που έχει επιλεγεί.
        const bool pq = cfg.method == "ivfpq";
        if (pq) {
            title = "IVFPQ";
        } else {
            title = "IVFFlat";
        }
        //δημιουργία του αντικειμένου για το ευρετήριο IVF.
        auto index = std::make_shared<IvfIndex>();
        bool cached = false;
        double seconds = 0.0;
        //φώρτωση ή κατασκευή του ευρετηρίου IVF.
        if (!prepareIvf(cfg, db, *index, cached, seconds, error)) {
            return false;
        }
        //εμφάνιση πληροφοριών για την προετοιμασια του ευρετηριου IVF.
        std::cout << title << " index (kclusters=" << index->coarse.rows << ", nprobe=" << cfg.nprobe << "): ";
        if (cached) {
            std::cout << "loaded from cache";
        } else {
            std::cout << "trained and saved";
        }
        std::cout << " in " << seconds << " s\n";
        retrieve = [index, &db, pq, nprobe = cfg.nprobe](const Matrix& q, int C) {
            return pq ? ivfPqSearch(*index, q.row(0), nprobe, C) : ivfFlatSearch(*index, db, q.row(0), nprobe, C);
        };
    }
    return true;
}