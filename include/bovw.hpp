#pragma once

#include "kmeans.hpp"

/*
    Bag of Visual Words μιας εικονας: x[j] = h_j / n, οπου n ειναι οι
    SIFT descriptors (γραμμες του `descriptors`, Matrix N x d) και h_j ποσοι
    απο αυτους εχουν centroid j (γραμμη του `vocabulary`, Matrix K x d) ως
    κοντινοτερο centroid (squared Euclidean distance, μικροτερο index σε
    ισοπαλια). Επιστρεφει Matrix 1 x K με μη αρνητικες τιμες που
    αθροιζουν σε 1.

    Πεταει std::invalid_argument αν οι διαστασεις δεν
    ταιριαζουν, αν υπαρχει μη πεπερασμενη τιμη, ή αν δεν υπαρχουν descriptors (n
    = 0 δεν εχει normalized histogram: ο caller πρεπει να αποφασισει τι θα κανει
    με την εικονα).
*/

Matrix computeBovw(const Matrix& descriptors, const Matrix& vocabulary);
