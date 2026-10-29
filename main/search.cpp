#include <cmath>
#include <fstream>
#include <iostream>

#include "args.hpp"
#include "bovw.hpp"
#include "database.hpp"
#include "evaluation.hpp"
#include "exact.hpp"
#include "hypercube.hpp"
#include "ivf.hpp"
#include "lsh.hpp"
#include "sift_extract.hpp"
#include "vocabulary.hpp"

int main(int argc, char* argv[]) {
	Config cfg;
	std::string error;
	// Ανάγνωση και έλεγχος των παραμέτρων εκτέλεσης
	if (!parseArgs(std::vector<std::string>(argv + 1, argv + argc), cfg,
	               error)) {
		std::cerr << "error: " << error << '\n';
		return 2;
	}
	// Έλεγχος ότι η επιλεγμένη μέθοδος υποστηρίζεται
	if (cfg.method != "exact" && cfg.method != "lsh" &&
	    cfg.method != "hypercube" && cfg.method != "ivfflat" &&
	    cfg.method != "ivfpq") {
		std::cerr << "error: unsupported method " << cfg.method << '\n';
		return 1;
	}
	// Προετοιμασία των δεδομένων και του vocabulary
	SearchData data;
	VocabInfo vocab;
	if (!prepareSearchData(cfg, data, vocab, error)) {
		std::cerr << "error: " << error << '\n';
		return 1;
	}
	// Εμφάνιση πληροφοριών για το vocabulary και τα δεδομένα
	std::cout << "Vocabulary K=" << cfg.vocab << ": "
	          << (vocab.fromCache ? "loaded from cache" : "trained and saved")
	          << " in " << vocab.seconds << " s\n";
	std::cout << "Queries: " << data.queries.rows << " (" << cfg.set
	          << "), database: " << data.database.rows << " images\n";
	std::cout << "BoVW vectors: " << data.imagesFromCache << " from cache, "
	          << data.imagesComputed << " computed with SIFT\n";
	RunResult run;
	std::string title;
	try {
		// Έναρξη μέτρησης του χρόνου κατασκευής του ευρετηρίου
		const auto start = std::chrono::steady_clock::now();
		if (cfg.method == "exact") {
			// Ακριβής αναζήτηση σε ολόκληρη τη βάση
			title = "Exact Search";
			run = runQueries(
			    data,
			    [&](const Matrix& query) {
				    return exactSearch(data.database, query);
			    },
			    cfg.repeat);
		} else if (cfg.method == "lsh") {
			// Κατασκευή του ευρετηρίου LSH
			title = "LSH";
			const LshIndex index =
			    buildLsh(data.database, cfg.k, cfg.L, cfg.w, cfg.seed);
			std::cout << "LSH index (k=" << cfg.k << ", L=" << cfg.L
			          << ", w=" << cfg.w << ", table size " << index.tableSize
			          << ") built in "
			          << std::chrono::duration<double>(
			                 std::chrono::steady_clock::now() - start)
			                 .count()
			          << " s\n";
			// Εκτέλεση αναζήτησης με LSH
			run = runQueries(
			    data,
			    [&](const Matrix& query) {
				    return lshSearch(index, data.database, query);
			    },
			    cfg.repeat);
		} else if (cfg.method == "ivfflat" || cfg.method == "ivfpq") {
			// Επιλογή μεταξύ IVFFlat και IVFPQ
			const bool pq = cfg.method == "ivfpq";
			title = pq ? "IVFPQ" : "IVFFlat";
			// Κατασκευή ή φόρτωση του ευρετηρίου IVF
			IvfIndex index;
			bool cached = false;
			double seconds = 0.0;
			if (!prepareIvf(cfg, data.database, index, cached, seconds,
			                error)) {
				std::cerr << "error: " << error << '\n';
				return 1;
			}
			// Εμφάνιση πληροφοριών του ευρετηρίου IVF
			std::cout << title << " index (kclusters=" << index.coarse.rows
			          << ", nprobe=" << cfg.nprobe;
			if (pq) {
				std::cout << ", M=" << cfg.M << ", nbits=" << cfg.nbits;
			}
			std::cout << "): "
			          << (cached ? "loaded from cache" : "trained and saved")
			          << " in " << seconds << " s\n";
			// Εκτέλεση αναζήτησης με IVFFlat ή IVFPQ
			run = runQueries(
			    data,
			    [&](const Matrix& query) {
				    return pq ? ivfPqSearch(index, query.row(0), cfg.nprobe)
					          : ivfFlatSearch(index, data.database,
					                          query.row(0), cfg.nprobe);
			    },
			    cfg.repeat);
		} else {
			// Κατασκευή του ευρετηρίου Hypercube
			title = "Hypercube";
			const HypercubeIndex index =
			    buildHypercube(data.database, cfg.kproj, cfg.w, cfg.seed);
			std::cout << "Hypercube index (kproj=" << cfg.kproj
			          << ", w=" << cfg.w << ", M=" << cfg.M
			          << ", probes=" << cfg.probes << ") built in "
			          << std::chrono::duration<double>(
			                 std::chrono::steady_clock::now() - start)
			                 .count()
			          << " s, " << index.vertices.size()
			          << " non-empty vertices\n";
			// Εκτέλεση αναζήτησης με Hypercube
			run = runQueries(
			    data,
			    [&](const Matrix& query) {
				    return hypercubeSearch(index, data.database, query, cfg.M,
					                       cfg.probes);
			    },
			    cfg.repeat);
		}
	} catch (const std::exception& e) {
		// Διαχείριση σφαλμάτων κατά την κατασκευή και την αναζήτηση
		std::cerr << "error: " << e.what() << '\n';
		return 1;
	}
	// Καταμέτρηση των queries που επέστρεψαν λιγότερα από 10 αποτελέσματα
	int fewer = 0;
	for (const auto& top : run.ranked) {
		if (static_cast<int>(top.size()) < kTopK) {
			++fewer;
		}
	}
	// Αποθήκευση των αποτελεσμάτων αναζήτησης
	if (!writeResults(cfg.out, title, data, run, error)) {
		std::cerr << "error: " << error << '\n';
		return 1;
	}
	// Αποθήκευση των χρόνων, εφόσον έχει δοθεί αρχείο timing
	if (!cfg.timing.empty()) {
		if (!writeTiming(cfg.timing, data, run, error)) {
			std::cerr << "error: " << error << '\n';
			return 1;
		}
	}
	// Εμφάνιση των συνολικών μετρικών αξιολόγησης
	std::cout << "Mean Recall@5: " << run.mean.recall5 << '\n';
	std::cout << "Mean Recall@10: " << run.mean.recall10 << '\n';
	std::cout << "mAP@10: " << run.mean.ap10 << '\n';

	// Εμφάνιση του μέσου χρόνου αναζήτησης
	std::cout << "Average query time: " << run.meanMs << " ms (mean of "
	          << run.repMs.size() << " repetition(s) of the query set)\n";
	// Εμφάνιση των queries με λιγότερα από 10 αποτελέσματα
	std::cout << "Queries with fewer than " << kTopK << " results: " << fewer
	          << " of " << data.queries.rows << '\n';
	std::cout << "Results written to " << cfg.out << '\n';
	return 0;
}