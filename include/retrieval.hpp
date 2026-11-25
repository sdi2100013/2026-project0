#pragma once

#include <functional>
#include <string>
#include <vector>

#include "args.hpp"
#include "database.hpp"

//τίπος συνάρτησης για την ανάκτηση των C πρώτων υποψήφιων εικόνων.
//επιστρέφει τους δείκτες των ηκόνων της βάσης με τη σειρά κατάταξής τους.
using TopC = std::function<std::vector<int>(const Matrix& query, int C)>;

//κατασκευή ή φώρτωση του ευρετηρίου ανάλογα με την επιλεγμένη μέθοδο αναζήτισης.
//Η βάση δεδομένων πρέπει να παραμένει διαθέσιμη όσο χρησιμοποιείτε η retrieve.
bool makeRetriever(const Config& cfg, const SearchData& data, TopC& retrieve, std::string& title, std::string& error);