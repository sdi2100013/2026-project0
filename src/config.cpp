#include "config.hpp"

#include <stdexcept>
#include <string>

Config parseArguments(int argc, char* argv[]) {
	Config config;

	for (int i = 1; i < argc; ++i) {
		std::string arg = argv[i];

		if (arg == "-hp") {
			if (i + 1 >= argc || argv[i + 1][0] == '-') {
				throw std::runtime_error("Missing value after -hp");
			}
			config.hpPath = argv[++i];
		}
		if (arg == "-mir") {
			if (i + 1 >= argc || argv[i + 1][0] == '-') {
				throw std::runtime_error("Missing value after -mir");
			}
			config.mirPath = argv[++i];
		}
		if (arg == "-split") {
			if (i + 1 >= argc || argv[i + 1][0] == '-') {
				throw std::runtime_error("Missing value after -split");
			}
			config.splitFile = argv[++i];
		}
		if (arg == "-set") {
			if (i + 1 >= argc|| argv[i + 1][0] == '-') {
				throw std::runtime_error("Missing value after -set");
			}
			config.dataSet = argv[++i];
		}
		if (arg == "-o") {
			if (i + 1 >= argc || argv[i + 1][0] == '-') {
				throw std::runtime_error("Missing value after -0");
			}
			config.outputFile = argv[++i];
		}
		if (arg == "-vocab") {
			if (i + 1 >= argc) {
				throw std::runtime_error("Missing value after -vocab");
			}
			config.vocab = std::stoi(argv[++i]);
		}
		if (arg == "-S") {
			if (i + 1 >= argc) {
				throw std::runtime_error("Missing value after -S");
			}
			config.S = std::stoi(argv[++i]);
		}
		if (arg == "-D") {
			if (i + 1 >= argc) {
				throw std::runtime_error("Missing value after -D");
			}
			config.D = std::stoi(argv[++i]);
		}
		if (arg == "-seed") {
			if (i + 1 >= argc) {
				throw std::runtime_error("Missing value after -seed");
			}
			config.seed = std::stoi(argv[++i]);
		}
		if (arg == "-exact") {
			if (config.method != Searchmethod::NONE) {
				throw std::runtime_error(
				    "More than one search method specidied");
			}
			config.method = Searchmethod::EXACT;
		}
		if (arg == "-lsh") {
			if (config.method != Searchmethod::NONE) {
				throw std::runtime_error(
				    "More than one search method specidied");
			}
			config.method = Searchmethod::LSH;
		}
		if (arg == "-hypercube") {
			if (config.method != Searchmethod::NONE) {
				throw std::runtime_error(
				    "More than one search method specidied");
			}
			config.method = Searchmethod::HYPERCUBE;
		}
		if (arg == "-ivfflat") {
			if (config.method != Searchmethod::NONE) {
				throw std::runtime_error(
				    "More than one search method specidied");
			}
			config.method = Searchmethod::IVFFLAT;
		}
		if (arg == "-ivfpq") {
			if (config.method != Searchmethod::NONE) {
				throw std::runtime_error(
				    "More than one search method specidied");
			}
			config.method = Searchmethod::IVFPQ;
		}
	}

	if (config.method == Searchmethod::NONE) {
		throw std::runtime_error("No search method specified");
	}

	// Έλεγχοσ παραμέτρων LSH
	if (config.method == Searchmethod::LSH) {
		for (int i = 1; i < argc; ++i) {
			std::string arg = argv[i];
			if (arg == "-k") {
				if (i + 1 >= argc) {
					throw std::runtime_error("Missing value after -k");
				}
				config.lsh.K = std::stoi(argv[++i]);
			}
			if (arg == "-L") {
				if (i + 1 >= argc) {
					throw std::runtime_error("Missing value after -l");
				}
				config.lsh.L = std::stoi(argv[++i]);
			}
			if (arg == "-w") {
				if (i + 1 >= argc) {
					throw std::runtime_error("Missing value after -w");
				}
				config.lsh.W = std::stof(argv[++i]);
			}
		}
	}

	// Έλεγχος παραμέτρων HYPERCUBE
	if (config.method == Searchmethod::HYPERCUBE) {
		for (int i = 1; i < argc; ++i) {
			std::string arg = argv[i];
			if (arg == "-kproj") {
				if (i + 1 >= argc) {
					throw std::runtime_error("Missing value after -kproj");
				}
				config.hypercube.kproj = std::stoi(argv[++i]);
			}
			if (arg == "-M") {
				if (i + 1 >= argc) {
					throw std::runtime_error("Missing value after -M");
				}
				config.hypercube.M = std::stoi(argv[++i]);
			}
			if (arg == "-w") {
				if (i + 1 >= argc) {
					throw std::runtime_error("Missing value after -w");
				}
				config.hypercube.W = std::stof(argv[++i]);
			}
			if (arg == "-probes") {
				if (i + 1 >= argc) {
					throw std::runtime_error("Missing value after -probes");
				}
				config.hypercube.probes = std::stoi(argv[++i]);
			}
		}
	}

	// Έλεγχος παραμέτρων IVFFLAT
	if (config.method == Searchmethod::IVFFLAT) {
		for (int i = 1; i < argc; ++i) {
			std::string arg = argv[i];
			if (arg == "-kclusters") {
				if (i + 1 >= argc) {
					throw std::runtime_error("Missing value after -kclusters");
				}
				config.ivfflat.kclusters = std::stoi(argv[++i]);
			}
			if (arg == "-nprobe") {
				if (i + 1 >= argc) {
					throw std::runtime_error("Missing value after -nprobe");
				}
				config.ivfflat.nprobe = std::stoi(argv[++i]);
			}
		}
	}

	// Έλεγχος παραμέτρων IVFPQ
	if (config.method == Searchmethod::IVFPQ) {
		for (int i = 1; i < argc; ++i) {
			std::string arg = argv[i];
			if (arg == "-kclusters") {
				if (i + 1 >= argc) {
					throw std::runtime_error("Missing value after -kclusters");
				}
				config.ivfpq.kclusters = std::stoi(argv[++i]);
			}
			if (arg == "-nprobe") {
				if (i + 1 >= argc) {
					throw std::runtime_error("Missing value after -nprobe");
				}
				config.ivfpq.nprobe = std::stoi(argv[++i]);
			}
			if (arg == "-M") {
				if (i + 1 >= argc) {
					throw std::runtime_error("Missing value after -M");
				}
				config.ivfpq.M = std::stoi(argv[++i]);
			}
			if (arg == "-nbits") {
				if (i + 1 >= argc) {
					throw std::runtime_error("Missing value after -nbits");
				}
				config.ivfpq.nbits = std::stoi(argv[++i]);
			}
		}
	}

	if (config.hpPath.empty()) {
		throw std::runtime_error("Missing required argument -hp");
	}
	if (config.mirPath.empty()) {
		throw std::runtime_error("Missing required argument -mir");
	}
	if (config.splitFile.empty()) {
		throw std::runtime_error("Missing required argument -split");
	}
	if (config.outputFile.empty()) {
		throw std::runtime_error("Missing required argument -o");
	}
	if (config.dataSet != "validation" && config.dataSet != "test") {
		throw std::runtime_error("-set must be either validation or test");
	}
	if (config.vocab <= 0) {
		throw std::runtime_error("-vocab must be positive");
	}
	if (config.S <= 0) {
		throw std::runtime_error("-S must be positive");
	}
	if (config.D <= 0) {
		throw std::runtime_error("-D must be poitive");
	}
	if (config.method == Searchmethod::LSH) {
		if (config.lsh.K <= 0 || config.lsh.L <= 0 || config.lsh.W <= 0.0) {
			throw std::runtime_error("Invalid LSH parameters");
		}
	}
	if (config.method == Searchmethod::HYPERCUBE) {
		if (config.hypercube.kproj <= 0 || config.hypercube.W <= 0.0 ||
		    config.hypercube.M <= 0 || config.hypercube.probes <= 0) {
			throw std::runtime_error("Invalid Hypercubfe parameters");
		}
	}
	if (config.method == Searchmethod::IVFFLAT) {
		if (config.ivfflat.kclusters == 0 || config.ivfflat.kclusters < -1 ||
		    config.ivfflat.nprobe <= 0) {
			throw std::runtime_error("Invalid IVFFlat parameters");
		}
	}
	if (config.method == Searchmethod::IVFPQ) {
		if (config.ivfpq.kclusters == 0 || config.ivfpq.kclusters < -1 ||
		    config.ivfpq.M <= 0 || config.ivfpq.nprobe <= 0 ||
		    config.ivfpq.nbits <= 0) {
			throw std::runtime_error("Invalid IVFPQ parameters");
		}
		if (config.vocab % config.ivfpq.M != 0) {
			throw std::runtime_error(
			    "-vocab must be divisible by -M for IVFPQ");
		}
	}

	for (int i = 1; i < argc; ++i) {
		std::string arg = argv[i];
		if (isCommonOption(arg) || isMethodOption(arg, config.method)) {
			if (i + 1 >= argc) {
				throw std::runtime_error("Missing value after " + arg);
			}

			++i;  // Skip value of argument
		} else if (arg == "-exact" || arg == "-lsh" || arg == "-hypercube" ||
		           arg == "-ivfflat" || arg == "-ivfpq") {
		} else {
			throw std::runtime_error("Unknown or invalid argument: " + arg);
		}
	}
	return config;
}

bool isCommonOption(const std::string& arg) {
	return arg == "-hp" || arg == "-mir" || arg == "-split" ||
	       arg == "-vocab" || arg == "-S" || arg == "-D" || arg == "-set" ||
	       arg == "-o" || arg == "-seed";
}

bool isMethodOption(const std::string& arg, Searchmethod method) {
	if (method == Searchmethod::LSH) {
		return arg == "-k" || arg == "-L" || arg == "-w";
	}
	if (method == Searchmethod::HYPERCUBE) {
		return arg == "-kproj" || arg == "-w" || arg == "-M" ||
		       arg == "-probes";
	}
	if (method == Searchmethod::IVFFLAT) {
		return arg == "-kclusters" || arg == "-nprobe";
	}
	if (method == Searchmethod::IVFPQ) {
		return arg == "-kclusters" || arg == "-nprobe" || arg == "-M" ||
		       arg == "-nbits";
	}

	// IF METHOD == EXACT
	return false;
}

