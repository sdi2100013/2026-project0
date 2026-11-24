#pragma once

#ifndef CONFIG_HPP
#define CONFIG_HPP

#include <iosfwd>
#include <string>
#include <vector>

// Παράμετροι εκτέλεσης του search.
struct Config {
	std::string hp;     // path του αρχείου hpatches
	std::string mir;    // path του αρχείου mirflickr
	std::string split;  // Path του αρχείου split
	std::string out;    // path του αρχείου εξόδου
	std::string set;    // Παίρνει τιμή "validation" ή "test"
	int vocab = 0;      // K, πλήθος οπτικών λέξεων
	int S = 500;  // μέγιστος αριθμός SIFT descriptors ανά εικόνα
	int D = 0;  // number of MIRFlickr images
	unsigned seed = 1;

	// Επιλογές για μέτρηση χρόνου
	int repeat = 1;
	std::string timing;

	std::string method;  // Παίρνει τιμές "exact", "lsh", "hypercube",
	                     // "ivfflat", "ivfpq"

	// Παράμετροι μεθόδων.
	int k = 4;          // lsh
	int L = 5;          // lsh
	double w = 0.05;    // lsh, hypercube
	int kproj = 10;     // hypercube
	int probes = 10;    // hypercube
	int M = 0;          // hypercube: υποψήφια σημεία, IVFPQ: υποδιανύσματα
	int kclusters = 0;  // ivfflat, αυτόματος υπολογισμός όταν είναι 0
	int nprobe = 5;     // ivfflat, ivfpq
	int nbits = 8;      // ivfpq

	// Παράμετροι του verify (γεωμετρική επαλήθευση)
	int C = 50;  // υποψήφιες εικόνες που επαληθεύονται (>= 10)
	double rho = 0.8;  // Lowe ratio test
	double tau = 3.0;  // κατώφλι επαναπροβολής σε pixels
	int T = 2000;      // μέγιστες επαναλήψεις RANSAC
	double p = 0.99;   // πιθανότητα επιτυχίας RANSAC
	bool adaptive = true;  // προσαρμοζόμενο πλήθος επαναλήψεων
};
// Επεξεργάζεται τα ορίσματα, εφαρμόζει τις προεπιλεγμένες τιμές
// και ελέγχει την εγκυρότητά τους.
// Σε περίπτωση σφάλματος επιστρέφει false και ενημερώνει το error.
// Με verify = true γίνονται δεκτά και τα -C, -rho, -tau, -T, -p, -adaptive, το
// -kclusters είναι υποχρεωτικό για IVF και τα -repeat, -timing δεν
// επιτρέπονται.
bool parseArgs(const std::vector<std::string>& args, Config& cfg,
               std::string& error, bool verify = false);

// Υπολογίζει το πλήθος των IVF clusters.
// Αν δεν έχει δοθεί kclusters, χρησιμοποιείται η ceil(sqrt(n)).
int resolveKclusters(const Config& cfg, int n);

// Εμφανίζει τις παραμέτρους εκτέλεσης.
void printConfig(const Config& cfg, std::ostream& os);

#endif