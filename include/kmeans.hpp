#pragma once

#include <vector>

// απλος dense row-major πινακας float
struct Matrix {
	int rows = 0, cols = 0;
	std::vector<float> data;  // οι τιμες των γραμμων διαδοχικα

	Matrix() = default;
	Matrix(int r, int c, float value = 0.f)
	    : rows(r), cols(c), data(static_cast<size_t>(r) * c, value) {}

	float* row(int i) { return &data[static_cast<size_t>(i) * cols]; }
	const float* row(int i) const {
		return &data[static_cast<size_t>(i) * cols];
	}
};

struct KMeansResult {
	Matrix centroids;               // K x d centroids
	std::vector<double> objective;  // J πριν και μετά από κάθε επανάληψη
	int iterations = 0;             // πόσες επαναλήψεις έγιναν
	bool converged = false;         // αν σταμάτησε πριν το maxIter
	int emptyRepaired = 0;          // διορθωμένα άδεια clusters
};

/*
    k-means++ αρχικοποιηση και μετα Lloyd iterations με squared Euclidean
    distance. Το data ειναι N*d και πρεπει να εχει πεπερασμενες τιμες.
    Ελαχιστοποιει το J. Σταματα οταν σταθεροποιηθουν οι αναθεσεις, οταν η
   σχετικη μειωση γινει <= tol ή οταν συμπληρωθουν maxIter iterations. Η
   τυχαιοτητα ελεγχεται απο το std::mt19937(seed). Σε αδειο cluster το centroid
   μεταφερεται στο πιο μακρινο διαθεσιμο σημειο. Αν ολα τα σημεια συμπιπτουν,
   παραμενει στη θεση του.
*/

// Απορριπτει μη εγκυρα K, δεδομενα, maxIter ή tol.
KMeansResult kmeans(const Matrix& data, int K, unsigned seed, int maxIter = 100,
                    double tol = 1e-4);

// Μονο Lloyd iterations απο τα δοσμενα K x d centroids.
// Το kmeans() προηγειται με k-means++ και κρατα τους ιδιους κανονες.
KMeansResult kmeansLloyd(const Matrix& data, const Matrix& initialCentroids,
                         int maxIter = 100, double tol = 1e-4);

// Επιστρεφει το κοντινοτερο centroid με squared Euclidean distance.
// σε ισοπαλια κερδιζει το μικροτερο index και προαιρετικα γραφεται η αποσταση.
int nearestCentroid(const float* x, const Matrix& centroids,
                    float* dist = nullptr);