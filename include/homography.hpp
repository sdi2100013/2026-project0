#pragma once

#include <Eigen/Dense>
#include <array>
#include <string>
#include <vector>

// Υπολογισμος ομογραφιας με Normalized DLT / SVD (Eigen, ολα σε double).
// Κατευθυνση παντου: query → candidate, p' ~ H p.
//
// Συμβασεις: η H ειναι Eigen::Matrix3d. Οριζεται μεχρι κλιμακα· η
// estimateHomography επιστρεφει H με νορμα Frobenius 1 και θετικο το στοιχειο
// με το μεγαλυτερο μετρο (δεν αλλαζει την απεικονιση).

struct Point2 {
	double x = 0, y = 0;
};

// p = σημειο στην query, c = αντιστοιχο σημειο στην candidate.
struct PointCorrespondence {
	Point2 p;
	Point2 c;
};

// ---- Εκφυλισμος ----
// Τετραδα αντιστοιχισσεων που δεν προσδιοριζει σταθερα ομογραφια: σε μια απο τις
// δυο εικονες υπαρχουν μη πεπερασμενες τιμες, επαναλαμβανομενα σημεια η
// (σχεδον) συνευθειακη τριαδα, δηλαδη 2*εμβαδον/(μεγαλυτερη πλευρα)^2 < 0.01. Ο
// ελεγχος γινεται ανεξαρτητα σε καθε εικονα και σε κανονικοποιημενες
// συντεταγμενες, αρα δεν εξαρταται απο την κλιμακα.
bool isDegenerateSample(const std::array<PointCorrespondence, 4>& sample);

// ---- Εκτιμηση ----
// Normalized DLT για m >= 4 αντιστοιχισεις (m = 4 η refit με m > 4): καθε
// συνολο σημειων μετασχηματιζεται με T = [s 0 -s*cx; 0 s -s*cy; 0 0 1] (κεντρο
// βαρους στο (0,0), μεση ευκλειδεια αποσταση sqrt(2)), η λυση του A h = 0 ειναι
// το δεξιο ιδιαζον διανυσμα της μικροτερης ιδιαζουσας τιμης και H = inverse(T')
// * Hn * T. Αποτυχια (false, μηνυμα στο error) αν: m < 4, μη πεπερασμενες
// τιμες, αποτυχια κανονικοποιησης, σχεδον συνευθειακο συνολο σε μια εικονα, A
// με ελλιπη βαθμιδα (δεν προσδιοριζεται μοναδικη H), η μη εγκυρη H.
bool estimateHomography(const std::vector<PointCorrespondence>& correspondences,
                        Eigen::Matrix3d& H, std::string& error);

// ---- Εγκυροτητα και προβολη ----
// Πεπερασμενα στοιχεια, μη μηδενικη και αριθμητικα αντιστρεψιμη:
// σ_min(H) / σ_max(H) > 1e-12 (ανεξαρτητο απο την κλιμακα της H).
bool isValidHomography(const Eigen::Matrix3d& H);
// Προβολη (a/c, b/c). false αν η H δεν ειναι εγκυρη, ο παρονομαστης c ειναι
// αριθμητικα μηδενικος (|c| <= 1e-12 * ||(a,b,c)||, ανεξαρτητο απο κλιμακα)
// η το αποτελεσμα δεν ειναι πεπερασμενο.
bool projectPoint(const Eigen::Matrix3d& H, const Point2& p, Point2& projected);
// Σφαλματα επαναπροβολης e = ||project(H, p) - c|| σε pixels της candidate για
// καθε αντιστοιχιση: η H ελεγχεται ΜΙΑ φορα (false αν ειναι ακυρη) και για καθε
// σημειο ελεγχεται μονο ο παρονομαστης, ωστε να μην εκτελειται SVD 3x3 ανα
// σημειο. Ακυρη προβολη δινει σφαλμα +απειρο (ποτε inlier, ποτε 0).
bool reprojectionErrors(const Eigen::Matrix3d& H,
                        const std::vector<PointCorrespondence>& correspondences,
                        std::vector<double>& errors);
