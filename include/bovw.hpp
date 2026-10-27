#pragma once

#include <opencv2/core.hpp>

/*
    Bag of Visual Words μιας εικονας: x[j] = h_j / n, οπου n ειναι οι
    SIFT descriptors (γραμμες του `descriptors`, N x d, CV_32F) και h_j ποσοι
    απο αυτους εχουν centroid j (γραμμη του `vocabulary`, K x d, CV_32F) ως
    κοντινοτερο centroid (squared Euclidean distance, μικροτερο index σε
    ισοπαλια). Επιστρεφει vector 1 x K CV_32F με μη αρνητικες τιμες που
    αθροιζουν σε 1.

    Πεταει std::invalid_argument αν οι τυποι ή οι διαστασεις δεν
    ταιριαζουν, αν υπαρχει μη πεπερασμενη τιμη, ή αν δεν υπαρχουν descriptors (n
    = 0 δεν εχει normalized histogram: ο caller πρεπει να αποφασισει τι θα κανει
    με την εικονα).
*/

cv::Mat computeBovw(const cv::Mat& descriptors, const cv::Mat& vocabulary);
