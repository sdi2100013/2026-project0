#include "homography.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace {

// Κατωφλια (ολα αδιαστατα, ωστε να μην εξαρτωνται απο την κλιμακα).
constexpr double kMinSpread = 1e-12;  // μεση αποσταση / κλιμακα συντεταγμενων
constexpr double kMinTriangleRatio = 1e-2;  // 2*εμβαδον / (μεγαλυτερη πλευρα)^2
constexpr double kMinSetSpreadRatio = 1e-4;  // μικρος / μεγαλος αξονας συνολου
constexpr double kMinRankRatio = 1e-8;       // σ_8(A) / σ_1(A)
constexpr double kMinSingularRatio = 1e-12;  // σ_min(H) / σ_max(H)
constexpr double kMinDenominator = 1e-12;    // |c| / ||(a,b,c)||

// Ενα σημειο ειναι εγκυρο μονο αν και οι δυο συντεταγμενες ειναι πεπερασμενες
// (οχι NaN η απειρο).
bool finite(const Point2& p) {
	return std::isfinite(p.x) && std::isfinite(p.y);
}

// Κανονικοποιημενο σημειο: T * (x, y, 1).
// Ο T εχει τη μορφη [s 0 tx; 0 s ty; 0 0 1], αρα αρκουν τα στοιχεια (0,0),
// (0,2), (1,1), (1,2) και δεν χρειαζεται πληρης πολλαπλασιασμος πινακων.
Point2 normalized(const Eigen::Matrix3d& T, const Point2& p) {
	return {T(0, 0) * p.x + T(0, 2), T(1, 1) * p.y + T(1, 2)};
}

// Μικρος προς μεγαλος αξονας (τυπικες αποκλισεις) του νεφους σημειων, 0..1.
// 0 σημαινει οτι ολα τα σημεια βρισκονται σε ευθεια.
double spreadRatio(const std::vector<Point2>& points,
                   const Eigen::Matrix3d& T) {
	// Αθροισμα των xx, yy, xy των κανονικοποιημενων σημειων (πινακας
	// συνδιακυμανσης, χωρις διαιρεση με το πληθος, που απλοποιειται στον λογο).
	double sxx = 0, syy = 0, sxy = 0;
	for (const Point2& p : points) {
		const Point2 n = normalized(T, p);  // κεντρο βαρους στο (0,0)
		sxx += n.x * n.x;
		syy += n.y * n.y;
		sxy += n.x * n.y;
	}
	// Ο πινακας συνδιακυμανσης 2x2 ειναι συμμετρικος.
	Eigen::Matrix2d cov;
	cov << sxx, sxy, sxy, syy;
	// Οι ιδιοτιμες του δινουν το τετραγωνο των δυο αξονων του νεφους,
	// σε αυξουσα σειρα.
	Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> eig(cov);
	// Προστασια απο μικρες αρνητικες τιμες λογω στρογγυλοποιησης.
	const double small = std::max(eig.eigenvalues()(0), 0.0);
	const double large = eig.eigenvalues()(1);
	// Ο λογος των αξονων ειναι η τετραγωνικη ριζα του λογου των ιδιοτιμων.
	return large > 0 ? std::sqrt(small / large) : 0.0;
}

// Προβολη με ηδη επικυρωμενη H: ελεγχει μονο τον παρονομαστη και το αποτελεσμα.
bool projectChecked(const Eigen::Matrix3d& H, const Point2& p,
                    Point2& projected) {
	if (!finite(p)) return false;
	// Ομογενεις συντεταγμενες του αποτελεσματος: H * (x, y, 1).
	const double a = H(0, 0) * p.x + H(0, 1) * p.y + H(0, 2);
	const double b = H(1, 0) * p.x + H(1, 1) * p.y + H(1, 2);
	const double c = H(2, 0) * p.x + H(2, 1) * p.y + H(2, 2);
	// Το μετρο του (a, b, c) χρησιμοποιειται για να κριθει ο παρονομαστης
	// ανεξαρτητα απο την κλιμακα της H.
	const double norm = std::sqrt(a * a + b * b + c * c);
	// Ο παρονομαστης c δεν πρεπει να ειναι (σχεδον) μηδεν σε σχεση με το norm.
	if (!std::isfinite(norm) || !(std::fabs(c) > kMinDenominator * norm))
		return false;
	// Μετατροπη απο ομογενεις σε καρτεσιανες συντεταγμενες.
	projected = {a / c, b / c};
	return finite(projected);
}

// Μετασχηματισμος κανονικοποιησης T = [s 0 -s*cx; 0 s -s*cy; 0 0 1]: (cx, cy)
// το κεντρο βαρους, s = sqrt(2) / (μεση ευκλειδεια αποσταση απο το κεντρο
// βαρους). Μετα απο αυτον το κεντρο βαρους ειναι (0,0) και η μεση αποσταση
// sqrt(2). false αν δεν υπαρχουν σημεια, υπαρχουν μη πεπερασμενες τιμες η ολα
// τα σημεια ταυτιζονται.
bool normalizationMatrix(const std::vector<Point2>& points,
                         Eigen::Matrix3d& T) {
	if (points.empty()) return false;
	// Κεντρο βαρους: μεσος ορος των συντεταγμενων.
	double cx = 0, cy = 0;
	for (const Point2& p : points) {
		if (!finite(p)) return false;
		cx += p.x;
		cy += p.y;
	}
	cx /= points.size();
	cy /= points.size();
	// Μεση ευκλειδεια αποσταση των σημειων απο το κεντρο βαρους.
	double meanDistance = 0;
	for (const Point2& p : points)
		meanDistance += std::hypot(p.x - cx, p.y - cy);
	meanDistance /= points.size();
	// Η εξαπλωση συγκρινεται με την κλιμακα των συντεταγμενων, ωστε το
	// κατωφλι να μην εξαρταται απο το μεγεθος της εικονας.
	const double scale = std::max({1.0, std::fabs(cx), std::fabs(cy)});
	if (!(meanDistance > kMinSpread * scale)) return false;
	// Παραγοντας κλιμακας ωστε η μεση αποσταση να γινει sqrt(2).
	const double s = std::sqrt(2.0) / meanDistance;
	T << s, 0, -s * cx, 0, s, -s * cy, 0, 0, 1;
	return std::isfinite(s);
}

// Τετραδα σημειων μιας εικονας που δεν προσδιοριζει ομογραφια: μη πεπερασμενα η
// ταυτιζομενα σημεια, η (σχεδον) συνευθειακη τριαδα. Για καθε τριαδα
// συγκρινεται το 2*εμβαδον με το τετραγωνο της μεγαλυτερης πλευρας, σε
// κανονικοποιημενες συντεταγμενες (αδιαστατο μεγεθος). Δυο ιδια σημεια δινουν
// εμβαδον 0 σε καθε τριαδα που τα περιεχει.
bool isDegenerateQuad(const std::array<Point2, 4>& points) {
	const std::vector<Point2> pts(points.begin(), points.end());
	// Η κανονικοποιηση αποτυγχανει οταν τα σημεια ταυτιζονται η δεν ειναι
	// πεπερασμενα: τοτε η τετραδα ειναι εκφυλισμενη.
	Eigen::Matrix3d T;
	if (!normalizationMatrix(pts, T)) return true;
	// Κανονικοποιημενες συντεταγμενες των τεσσαρων σημειων.
	std::array<Point2, 4> n;
	for (int i = 0; i < 4; ++i) n[i] = normalized(T, points[i]);
	// Ελεγχουμε και τις 4 δυνατες τριαδες (i, j, k).
	for (int i = 0; i < 4; ++i) {
		for (int j = i + 1; j < 4; ++j) {
			for (int k = j + 1; k < 4; ++k) {
				// Διπλασιο εμβαδον του τριγωνου (μετρο του εξωτερικου
				// γινομενου).
				const double cross =
				    std::fabs((n[j].x - n[i].x) * (n[k].y - n[i].y) -
					          (n[j].y - n[i].y) * (n[k].x - n[i].x));
				// Μηκος της μεγαλυτερης πλευρας του τριγωνου.
				const double longest =
				    std::max({std::hypot(n[j].x - n[i].x, n[j].y - n[i].y),
					          std::hypot(n[k].x - n[i].x, n[k].y - n[i].y),
					          std::hypot(n[k].x - n[j].x, n[k].y - n[j].y)});
				// Αν longest == 0 τα τρια σημεια ταυτιζονται. Αλλιως, αν ο
				// λογος ειναι πολυ μικρος, τα σημεια ειναι σχεδον συνευθειακα.
				if (longest == 0 ||
				    cross < kMinTriangleRatio * longest * longest)
					return true;
			}
		}
	}
	return false;
}

}  // namespace

// Μια τετραδα αντιστοιχισεων ειναι εκφυλισμενη αν ειναι εκφυλισμενα τα σημεια
// της query η τα σημεια της candidate: ο ελεγχος γινεται ανεξαρτητα σε καθε
// εικονα.
bool isDegenerateSample(const std::array<PointCorrespondence, 4>& sample) {
	std::array<Point2, 4> query, candidate;
	for (int i = 0; i < 4; ++i) {
		query[i] = sample[i].p;
		candidate[i] = sample[i].c;
	}
	return isDegenerateQuad(query) || isDegenerateQuad(candidate);
}

// Normalized DLT: εκτιμα την ομογραφια query → candidate απο m >= 4
// αντιστοιχισεις.
bool estimateHomography(const std::vector<PointCorrespondence>& correspondences,
                        Eigen::Matrix3d& H, std::string& error) {
	const size_t m = correspondences.size();
	// Η ομογραφια εχει 8 βαθμους ελευθεριας και καθε αντιστοιχια δινει 2
	// εξισωσεις.
	if (m < 4) {
		error = "at least 4 correspondences are required";
		return false;
	}
	// Χωριζουμε τα σημεια των δυο εικονων και ελεγχουμε οτι ειναι πεπερασμενα.
	std::vector<Point2> query, candidate;
	for (const PointCorrespondence& pc : correspondences) {
		if (!finite(pc.p) || !finite(pc.c)) {
			error = "non-finite point coordinates";
			return false;
		}
		query.push_back(pc.p);
		candidate.push_back(pc.c);
	}

	// 1. Ανεξαρτητη κανονικοποιηση των δυο συνολων: T (query), Tp (candidate).
	Eigen::Matrix3d T, Tp;
	if (!normalizationMatrix(query, T) || !normalizationMatrix(candidate, Tp)) {
		error = "normalization failed (coincident points)";
		return false;
	}
	// Αν ολα τα σημεια μιας εικονας ειναι σχεδον σε ευθεια, η ομογραφια δεν
	// προσδιοριζεται.
	if (spreadRatio(query, T) < kMinSetSpreadRatio ||
	    spreadRatio(candidate, Tp) < kMinSetSpreadRatio) {
		error = "points are (nearly) collinear in one of the images";
		return false;
	}

	// 2. Πινακας A (2m x 9) απο τις κανονικοποιημενες συντεταγμενες:
	//    [-x -y -1  0  0  0  u*x u*y u]
	//    [ 0  0  0 -x -y -1  v*x v*y v]
	Eigen::MatrixXd A(2 * m, 9);
	for (size_t k = 0; k < m; ++k) {
		const Point2 a = normalized(T, query[k]);       // (x, y)
		const Point2 b = normalized(Tp, candidate[k]);  // (u, v)
		// Δυο γραμμες ανα αντιστοιχια: μια για την u και μια για την v.
		A.row(2 * k) << -a.x, -a.y, -1, 0, 0, 0, b.x * a.x, b.x * a.y, b.x;
		A.row(2 * k + 1) << 0, 0, 0, -a.x, -a.y, -1, b.y * a.x, b.y * a.y, b.y;
	}

	// 3. SVD: A = U Sigma V^T. Λυση h = τελευταια στηλη του V (μικροτερη σ).
	// Χρειαζομαστε μονο το V, γι' αυτο ζηταμε μονο το ComputeFullV.
	Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeFullV);
	const Eigen::VectorXd sigma = svd.singularValues();
	// Μοναδικη λυση ⇔ rank(A) = 8 ⇔ η 8η ιδιαζουσα τιμη δεν ειναι αμελητεα.
	if (!(sigma(0) > 0) || !(sigma(7) > kMinRankRatio * sigma(0))) {
		error = "insufficient geometry: the DLT system has rank < 8";
		return false;
	}
	// Το δεξιο ιδιαζον διανυσμα της μικροτερης σ ελαχιστοποιει το ||A h||.
	const Eigen::VectorXd h = svd.matrixV().col(8);

	// 4. h → Hn (σειρα γραμμων) και επαναφορα: H = inverse(Tp) * Hn * T.
	Eigen::Matrix3d Hn;
	Hn << h(0), h(1), h(2), h(3), h(4), h(5), h(6), h(7), h(8);
	H = Tp.inverse() * Hn * T;
	if (!H.allFinite()) {
		error = "non-finite homography";
		return false;
	}

	// 5. Κλιμακα και προσημο: νορμα 1, θετικο το μεγαλυτερο στοιχειο (η
	//    απεικονιση δεν αλλαζει).
	H /= H.norm();
	Eigen::Index row, col;
	H.cwiseAbs().maxCoeff(&row, &col);
	if (H(row, col) < 0) H = -H;

	// Τελικος ελεγχος: η H πρεπει να ειναι αριθμητικα αντιστρεψιμη.
	if (!isValidHomography(H)) {
		error = "the estimated homography is not valid (singular)";
		return false;
	}
	return true;
}

// Μια ομογραφια ειναι εγκυρη αν ειναι πεπερασμενη και αριθμητικα αντιστρεψιμη.
bool isValidHomography(const Eigen::Matrix3d& H) {
	if (!H.allFinite()) return false;
	Eigen::JacobiSVD<Eigen::Matrix3d> svd(H);  // σ(0) >= σ(1) >= σ(2) >= 0
	const Eigen::Vector3d s = svd.singularValues();
	// Ο λογος μικροτερης προς μεγαλυτερη ιδιαζουσα τιμη δεν εξαρταται απο την
	// κλιμακα της H, σε αντιθεση με μια απολυτη τιμη της οριζουσας.
	return s(0) > 0 && s(2) > kMinSingularRatio * s(0);
}

// Προβολη ενος σημειου με ελεγχο και της εγκυροτητας της H.
bool projectPoint(const Eigen::Matrix3d& H, const Point2& p,
                  Point2& projected) {
	return isValidHomography(H) && projectChecked(H, p, projected);
}

// Σφαλματα επαναπροβολης για ολες τις αντιστοιχισεις με μια μονο αξιολογηση της
// H.
bool reprojectionErrors(const Eigen::Matrix3d& H,
                        const std::vector<PointCorrespondence>& correspondences,
                        std::vector<double>& errors) {
	errors.clear();
	if (!isValidHomography(H)) return false;  // μια φορα ανα H, οχι ανα σημειο
	errors.reserve(correspondences.size());
	for (const PointCorrespondence& pc : correspondences) {
		// Που προβλεπει η H οτι θα βρισκεται το σημειο της query στην
		// candidate.
		Point2 predicted;
		double error =
		    std::numeric_limits<double>::infinity();  // ακυρη προβολη
		if (projectChecked(H, pc.p, predicted)) {
			// Ευκλειδεια αποσταση απο την πραγματικη θεση στην candidate.
			const double e =
			    std::hypot(predicted.x - pc.c.x, predicted.y - pc.c.y);
			if (std::isfinite(e)) error = e;
		}
		// Ακυρη προβολη = απειρο σφαλμα, αρα ποτε inlier.
		errors.push_back(error);
	}
	return true;
}