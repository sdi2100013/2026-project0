#pragma once

#ifndef CONFIG_HPP
#define CONFIG_HPP

#include <string>

//Δημιουργείτε μεταβλητή Searchmethod με enumeration για να μπορεί αν παρθεί συγκεκριμένο όνομα για την μέθοδο εκτέλεσης
enum class Searchmethod{
    NONE,
    EXACT,
    LSH,
    HYPERCUBE,
    IVFFLAT,
    IVFPQ
};

//Κάθε μέθοδος εκτέλεσης έχει διαφορετικό πλήρθος παραμέτρων οπότε φτιάχνουμε ξεχωριστά struct για κάθε μέθοδο.
//Δίνεται μαι πρώτη αρχικοποίηση καθώς αν δεν δωθεί από την είσοδο θα πρέπει να τρέχει με default τιμές.
struct LSHConfig{
    int K = 4;
    int L = 5;
    double W = 0.05;
};

struct HypercubeConfig{
    int kproj = 10;
    double W = 0.05;
    int M = 500;
    int probes = 10;
};

struct IVFFLATConfig{
    int kclusters = -1;
    int nprobe = 5;
};

struct IVFPQConfig{
    int kclusters = -1;
    int nprobe = 5;
    int M = 16;
    int nbits = 8;
};
//Η exact δεν εχει arguments στην εκφώνηση

struct Config{
    std::string hpPath;
    std::string mirPath;
    std::string splitFile;
    std::string outputFile;

    int vocab;
    int S = 500;
    int D;
    std::string dataSet;
    int seed;

    //Σε περίπτωση που ο χρήστης δεν επιλέξει κάποια μέθοδο
    Searchmethod method = Searchmethod::NONE;

    LSHConfig lsh;
    HypercubeConfig hypercube;
    IVFFLATConfig ivfflat;
    IVFPQConfig ivfpq; 
    
};

//Δέχεται τα argumnts από τον χρήστη 
Config parseArguments(int argc, char* argv[]);

#endif