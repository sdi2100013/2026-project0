#include "args.hpp"

#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <map>
#include <ostream>
#include <set>

namespace {

//Παράμετροι της γραμμής εντολών που απαιτούν τιμή
const std::set<std::string> kValueFlags = {"-hp", "-mir", "-split", "-vocab", "-S",     "-D",       "-set",
                                           "-o",  "-seed", "-k",     "-L",     "-w",     "-kproj",   "-M",
                                           "-probes", "-kclusters", "-nprobe", "-nbits", "-repeat", "-timing"};
//Μέθοδοι αναζήτησης
const std::set<std::string> kMethods = {"exact", "lsh", "hypercube", "ivfflat", "ivfpq"};

//Παράμετροι που γίνοτναι εποδεκτοί ανά μέθοδο
const std::map<std::string, std::set<std::string>> kOnlyFor = {
    {"-k", {"lsh"}},
    {"-L", {"lsh"}},
    {"-w", {"lsh", "hypercube"}},
    {"-kproj", {"hypercube"}},
    {"-probes", {"hypercube"}},
    {"-M", {"hypercube", "ivfpq"}},
    {"-kclusters", {"ivfflat", "ivfpq"}},
    {"-nprobe", {"ivfflat", "ivfpq"}},
    {"-nbits", {"ivfpq"}},
};

//Ματατροπή string σε int και εξετάζει αν είναι εντός ορίων
bool toInteger(const std::string& s, long long lo, long long hi, long long& out) {
    char* end = nullptr;
    errno = 0;
    out = std::strtoll(s.c_str(), &end, 10);
    return !s.empty() &&
			end == s.c_str() + s.size() &&
			errno == 0 && 
			out >= lo && 
			out <= hi;
}

//Μετατροπή string σε double και ελέγχει αν είναι έγκυρος
bool toDouble(const std::string& s, double& out) {
    char* end = nullptr;
    errno = 0;
    out = std::strtod(s.c_str(), &end);
    return !s.empty() && 
			end == s.c_str() + s.size() && 
			errno == 0 && 
			std::isfinite(out) && out > 0;
}
}//namespace

bool parseArgs(const std::vector<std::string>& args, Config& cfg, std::string& error) {
	//Επαναφροά των προεπιλεγμένων ρυθμίσεων
    cfg = Config();
	//Αποθύκευση των παραμέτρων και των αντίστοιχων τιμών τους
    std::map<std::string, std::string> given;  

    for (size_t i = 0; i < args.size(); ++i) {
        const std::string& a = args[i];
		bool isMethod = a.size() > 1 && a[0] == '-' && kMethods.count(a.substr(1));
        if (isMethod) {
            if (!cfg.method.empty()) {
                error = "only one search method allowed, got -" + cfg.method + " and " + a;
                return false;
            }
            cfg.method = a.substr(1);
        } else if (kValueFlags.count(a) > 0) {
			//Έλεγχοσ για διπλότυπα
            if (given.count(a) > 0) {
                error = "duplicate argument " + a;
                return false;
            }
            if (i + 1 >= args.size()) {
                error = "missing value for " + a;
                return false;
            }
            const std::string& v = args[++i];
			bool isMethodVal = (v.size() > 1 && v[0] == '-' && kMethods.count(v.substr(1)));
            if (kValueFlags.count(v) || isMethodVal) {
                error = "missing value for " + a + " (found " + v + ")";
                return false;
            }
            given[a] = v;
        } else {
            error = "unknown argument '" + a + "'";
            return false;
        }
    }

    if (cfg.method.empty()) {
        error = "a search method is required: -exact, -lsh, -hypercube, -ivfflat or -ivfpq";
        return false;
    }
    for (const auto& entry : given) {
        auto it = kOnlyFor.find(entry.first);
        if (it != kOnlyFor.end() && !it->second.count(cfg.method)) {
            error = entry.first + " is not valid with -" + cfg.method;
            return false;
        }
    }
	//Ελέγχος ότι δόθηκαν όλα όσα χρειαζόμαστε
    for (const char* requiredFlag : {"-hp", "-mir", "-split", "-vocab", "-D", "-set", "-o"}) {
        if (given.count(requiredFlag) == 0) {
            error = std::string("missing required argument ") + requiredFlag;
            return false;
        }
    }

    //Έλεγψοσ γισ ακέραιες μεταβλητές ότι είνια σε επιτρεπτά όρια 
    auto readInt = [&](const char* flag, int& result, long long lo, long long hi) {
        auto it = given.find(flag);
        if (it == given.end()){ 
			return true;
		}
        long long v;
        if (!toInteger(it->second, lo, hi, v)) {
            error = std::string("invalid value '") + it->second + "' for " + flag + " (integer in [" +
                    std::to_string(lo) + ", " + std::to_string(hi) + "] expected)";
            return false;
        }
        result = static_cast<int>(v);
        return true;
    };

	//Ανάθεση
    cfg.hp = given["-hp"];
    cfg.mir = given["-mir"];
    cfg.split = given["-split"];
    cfg.out = given["-o"];
    cfg.set = given["-set"];
    cfg.timing = given.count("-timing") ? given["-timing"] : "";
    if (cfg.set != "validation" && cfg.set != "test") {
        error = "invalid value '" + cfg.set + "' for -set (validation or test expected)";
        return false;
    }

    const long long kMax = 1000000000;
    long long seed = cfg.seed;
    if (given.count("-seed") && !toInteger(given["-seed"], 0, 4294967295LL, seed)) {
        error = "invalid value '" + given["-seed"] + "' for -seed (integer in [0, 4294967295] expected)";
        return false;
    }
    cfg.seed = static_cast<unsigned>(seed);

    if (!readInt("-vocab", cfg.vocab, 1, kMax) || !readInt("-S", cfg.S, 1, kMax) || !readInt("-D", cfg.D, 1, kMax) ||
        !readInt("-k", cfg.k, 1, kMax) || !readInt("-L", cfg.L, 1, kMax) || !readInt("-kproj", cfg.kproj, 1, 30) ||
        !readInt("-probes", cfg.probes, 1, kMax) || !readInt("-M", cfg.M, 1, kMax) ||
        !readInt("-kclusters", cfg.kclusters, 1, kMax) || !readInt("-nprobe", cfg.nprobe, 1, kMax) ||
        !readInt("-nbits", cfg.nbits, 1, 16) || !readInt("-repeat", cfg.repeat, 1, 1000))
        return false;

    if (given.count("-w") && !toDouble(given["-w"], cfg.w)) {
        error = "invalid value '" + given["-w"] + "' for -w (positive number expected)";
        return false;
    }

    if (!given.count("-M")){ 
		cfg.M = cfg.method == "ivfpq" ? 16 : 500;  // μόνο για hupercube και ivfpq
	}
	if (cfg.method == "ivfpq" && cfg.vocab % cfg.M != 0) {
        error = "-vocab (" + std::to_string(cfg.vocab) + ") must be divisible by -M (" + std::to_string(cfg.M) +
                ") for -ivfpq";
        return false;
    }
    return true;
}


//Υπολογίζει τον αριθμό των clusters όταν δεν έχει δωθεί από τον χρήστη
int resolveKclusters(const Config& cfg, int n) {
    if (cfg.kclusters > 0){
		return cfg.kclusters;
	}
	double root = std::sqrt(static_cast<double>(n));
	return static_cast<int>(std::ceil(root));
}

//Εμφάνιση ρυθμίσεων που θα γίνουν χρήση
void printConfig(const Config& c, std::ostream& os) {
    os << "hp=" << c.hp << " mir=" << c.mir << " split=" << c.split << " set=" << c.set << " o=" << c.out << "\n"
       << "vocab=" << c.vocab << " S=" << c.S << " D=" << c.D << " seed=" << c.seed << "\n"
       << "method=" << c.method;
    if (c.method == "lsh"){
		os << " k=" << c.k << " L=" << c.L << " w=" << c.w;
	}
    if (c.method == "hypercube"){
        os << " kproj=" << c.kproj << " w=" << c.w << " M(max candidates)=" << c.M << " probes=" << c.probes;
	}
    if (c.method == "ivfflat" || c.method == "ivfpq"){
        os << " kclusters=";
        if (c.kclusters > 0){
			os << c.kclusters; 
		}else{ 
			os << "auto(ceil(sqrt(n)))";
		}
        os << " nprobe=" << c.nprobe;
    }
    if (c.method == "ivfpq"){
		os << " M(sub-vectors)=" << c.M << " nbits=" << c.nbits;
	}
    os << "\n";
}