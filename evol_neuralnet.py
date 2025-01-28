#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <random>
#include <fstream>
#include <chrono>
#include <iomanip>
#include <filesystem>
#include "ga/ga.h"

// global flag to enable/disable debug messages
bool debug_mode = true;
// global file stream for logging evolution data
std::ofstream logFile;

/**
struct containing all configurable parameters for the genetic algorithm.
includes basic ga settings, convergence criteria, and boundary values
for all neural network hyperparameters that will be evolved.
**/
struct GAParameters {
    // basic genetic algorithm parameters
    int populationSize = 15;        // size of population in each generation
    int nGenerations = 5;           // maximum number of generations to evolve
    float pMutation = 0.2f;         // probability of mutation occurring
    float pCrossover = 0.7f;        // probability of crossover occurring

    // criteria for early stopping
    float stoppingThreshold = 0.92f;  // minimum average fitness to consider convergence
    int requiredConsecutiveGenerations = 5;  // number of generations that must meet threshold

    // boundaries for neural network hyperparameters
    float minLearningRate = 0.0001f;  // minimum allowed learning rate
    float maxLearningRate = 0.1f;     // maximum allowed learning rate
    float minDropoutRate = 0.0f;      // minimum allowed dropout rate
    float maxDropoutRate = 0.5f;      // maximum allowed dropout rate
    int minBatchSize = 16;            // minimum batch size for training
    int maxBatchSize = 128;           // maximum batch size for training
    int minEpochs = 1;                // minimum number of training epochs
    int maxEpochs = 50;               // maximum number of training epochs
    int minHiddenLayers = 1;          // minimum number of hidden layers
    int maxHiddenLayers = 5;          // maximum number of hidden layers
    int minLayerSize = 16;            // minimum nodes per hidden layer
    int maxLayerSize = 128;           // maximum nodes per hidden layer
};

// global instance of parameters used throughout the evolution process
GAParameters globalParams;

/**
validates and sets a float parameter within specified bounds.
logs an error message if the value is invalid or out of bounds.

@param param reference to the parameter to be set
@param value string containing the value to parse
@param minVal minimum allowed value for the parameter
@param maxVal maximum allowed value for the parameter
@param paramName name of the parameter for error reporting
@return true if parameter was set successfully, false otherwise
**/
bool setFloatParameter(float& param, const char* value, float minVal, float maxVal, const std::string& paramName) {
    try {
        float val = std::stof(value);
        if (val < minVal || val > maxVal) {
            std::cerr << "error: " << paramName << " must be between " << minVal << " and " << maxVal << std::endl;
            return false;
        }
        param = val;
        return true;
    } catch (const std::exception&) {
        std::cerr << "error: invalid value for " << paramName << std::endl;
        return false;
    }
}

/**
validates and sets an integer parameter within specified bounds.
logs an error message if the value is invalid or out of bounds.

@param param reference to the parameter to be set
@param value string containing the value to parse
@param minVal minimum allowed value for the parameter
@param maxVal maximum allowed value for the parameter
@param paramName name of the parameter for error reporting
@return true if parameter was set successfully, false otherwise
**/
bool setIntParameter(int& param, const char* value, int minVal, int maxVal, const std::string& paramName) {
    try {
        int val = std::stoi(value);
        if (val < minVal || val > maxVal) {
            std::cerr << "error: " << paramName << " must be between " << minVal << " and " << maxVal << std::endl;
            return false;
        }
        param = val;
        return true;
    } catch (const std::exception&) {
        std::cerr << "error: invalid value for " << paramName << std::endl;
        return false;
    }
}

/**
generates a unique log file path by creating a "logs" directory (if it doesn't exist)
and appending a timestamped filename.

@return a std::string representing the full path to the log file.
        the file path includes the "logs" directory and a filename
        in the format "evolution_log_<yyyy-mm-dd_hh-mm-ss>.csv".
**/
std::string getLogFilePath() {
    // create a logs directory if it doesn't exist
    const std::string logsDir = "../logs";
    if (!std::filesystem::exists(logsDir)) {
        std::filesystem::create_directory(logsDir);
    }

    // generate a unique filename using the current timestamp
    auto now = std::chrono::system_clock::now();
    std::time_t now_c = std::chrono::system_clock::to_time_t(now);
    std::tm now_tm = *std::localtime(&now_c);

    std::ostringstream oss;
    oss << logsDir << "/evolution_log_" << std::put_time(&now_tm, "%Y-%m-%d_%H-%M-%S") << ".csv";
    return oss.str();
}


/**
helper function to write the current evaluation result to the log file.
handles formatting and writing of all neural network parameters and fitness score.

@param params vector of hyperparameters being evaluated
@param num_hidden_layers number of active hidden layers
@param batch_size current batch size setting
@param epochs number of training epochs
@param activation_function index of activation function
@param fitness achieved fitness score
**/
void logEvaluationResult(const std::vector<float>& params,
                        int num_hidden_layers,
                        int batch_size,
                        int epochs,
                        int activation_function,
                        float fitness) {
    if (!logFile.is_open()) return;

    logFile << std::fixed << std::setprecision(8)
            << params[0] << "," << params[1] << ","
            << batch_size << "," << epochs << ","
            << activation_function << "," << num_hidden_layers << ",\"[";

    // write hidden layer sizes
    for (int i = 0; i < num_hidden_layers; ++i) {
        if (i > 0) logFile << ", ";
        logFile << static_cast<int>(params[6 + i]);
    }
    logFile << "]\"" << "," << fitness << "\n";
    logFile.flush();
}

/**
builds the json command string for the neural network evaluation script.
formats all hyperparameters into the required json structure.

@param learning_rate learning rate for the neural network
@param dropout_rate dropout rate for regularization
@param batch_size training batch size
@param epochs number of training epochs
@param activation_function index of activation function to use
@param num_hidden_layers number of hidden layers
@param hidden_sizes string containing formatted layer sizes
@return formatted command string for the python script
**/
std::string buildPythonCommand(float learning_rate,
                             float dropout_rate,
                             int batch_size,
                             int epochs,
                             int activation_function,
                             int num_hidden_layers,
                             const std::string& hidden_sizes) {
    std::ostringstream cmd;
    cmd << "python ../evol_neuralnet.py \""
        << R"({\"learning_rate\":)" << learning_rate
        << R"(, \"dropout_rate\":)" << dropout_rate
        << R"(, \"batch_size\":)" << batch_size
        << R"(, \"epochs\":)" << epochs
        << R"(, \"activation_function\":)" << activation_function
        << R"(, \"num_hidden_layers\":)" << num_hidden_layers
        << R"(, \"hidden_sizes\":)" << hidden_sizes
        << "}\"";
    return cmd.str();
}

/**
formats the hidden layer sizes into a json-compatible string.
validates that all layer sizes are positive before including them.

@param params vector containing all neural network parameters
@param num_hidden_layers number of active hidden layers to process
@return formatted string of layer sizes or empty string if validation fails
**/
std::string formatHiddenSizes(const std::vector<float>& params, int num_hidden_layers) {
    std::ostringstream hidden_sizes;
    hidden_sizes << "[";
    for (int i = 0; i < num_hidden_layers; ++i) {
        // validate layer size
        if (static_cast<int>(params[6 + i]) <= 0) {
            return "";  // invalid layer size detected
        }
        if (i > 0) {
            hidden_sizes << ",";
        }
        hidden_sizes << static_cast<int>(params[6 + i]);
    }
    hidden_sizes << "]";
    return hidden_sizes.str();
}

/**
executes a python script to evaluate the fitness of a neural network
based on provided hyperparameters.

@param params a vector of floats representing hyperparameters:
       - params[0]: learning rate
       - params[1]: dropout rate
       - params[2]: batch size
       - params[3]: number of epochs
       - params[4]: activation function
       - params[5]: number of hidden layers
       - params[6 + i]: size of the i-th hidden layer
@return a float value representing the fitness score of the neural network.
        returns 0.0f if evaluation fails or produces invalid results.
**/
float evaluate_nn(const std::vector<float>& params) {
    // extract and convert basic parameters
    float learning_rate = params[0];
    float dropout_rate = params[1];
    int batch_size = static_cast<int>(params[2]);
    int epochs = static_cast<int>(params[3]);
    int activation_function = static_cast<int>(params[4]);
    int num_hidden_layers = static_cast<int>(params[5]);

    // format hidden layer configuration
    std::string hidden_sizes = formatHiddenSizes(params, num_hidden_layers);
    if (hidden_sizes.empty()) {
        // invalid layer configuration detected
        logEvaluationResult(params, num_hidden_layers, batch_size, epochs,
                           activation_function, 0.0f);
        return 0.0f;
    }

    // build and execute python command
    std::string cmd = buildPythonCommand(learning_rate, dropout_rate, batch_size,
                                       epochs, activation_function, num_hidden_layers,
                                       hidden_sizes);

    if (debug_mode) {
        std::cout << "[training]: executing command: " << cmd << std::endl;
    }

    // execute python script and capture output
    FILE* pipe = _popen((cmd + " 2>&1").c_str(), "r");  // redirect stderr to stdout
    if (!pipe) {
        std::cerr << "error: could not open pipe to python script!" << std::endl;
        logEvaluationResult(params, num_hidden_layers, batch_size, epochs,
                           activation_function, 0.0f);
        return 0.0f;
    }

    // read script output (expecting single line with fitness score)
    char buffer[128];
    std::string result;
    if (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
        result = buffer;
    }

    // check script execution status
    int returnCode = _pclose(pipe);
    if (returnCode != 0) {
        std::cerr << "error: python script failed with return code " << returnCode << std::endl;
        logEvaluationResult(params, num_hidden_layers, batch_size, epochs,
                           activation_function, 0.0f);
        return 0.0f;
    }

    // parse and validate fitness score
    try {
        float fitness = std::stof(result);
        if (debug_mode) {
            std::cout << "[training]: fitness score: " << result << std::endl;
        }
        logEvaluationResult(params, num_hidden_layers, batch_size, epochs,
                           activation_function, fitness);
        return fitness;
    } catch (const std::exception& e) {
        std::cerr << "[error]: invalid output from python script: " << e.what() << std::endl;
        logEvaluationResult(params, num_hidden_layers, batch_size, epochs,
                           activation_function, 0.0f);
        return 0.0f;
    }
}


/**
objective function for galib. evaluates the fitness of a genome
by passing its parameters to the evaluate_nn function.
@param g a reference to the genome being evaluated
@return a float value representing the fitness score of the genome
**/
float Objective(GAGenome& g) {
    const auto& genome = dynamic_cast<GA1DArrayGenome<float>&>(g);

    // extract hyperparameters from the genome
    std::vector<float> params;
    for (int i = 0; i < genome.size(); ++i) {
        params.push_back(genome.gene(i));
    }

    // evaluate the fitness using the evaluate_nn function
    return evaluate_nn(params);
}


/**
custom genome initializer for galib. randomly generates initial values for all neural network
hyperparameters according to the bounds specified in globalParams.

each genome contains the following genes:
- gene[0]: learning rate (float)
- gene[1]: dropout rate (float)
- gene[2]: batch size (int)
- gene[3]: epochs (int)
- gene[4]: activation function (int, 0-3)
- gene[5]: number of hidden layers (int)
- gene[6+]: size of each hidden layer (int), with unused layers set to 0

@param g reference to the genome being initialized
**/
void customInitializer(GAGenome& g) {
    auto& genome = dynamic_cast<GA1DArrayGenome<float>&>(g);

    // set up random number generator
    std::random_device rd;
    std::mt19937 gen(rd());

    // create distributions for each parameter type according to globalParams bounds
    std::uniform_real_distribution<float> lr_dist(globalParams.minLearningRate, globalParams.maxLearningRate);
    std::uniform_real_distribution<float> dropout_dist(globalParams.minDropoutRate, globalParams.maxDropoutRate);
    std::uniform_int_distribution<int> batch_dist(globalParams.minBatchSize, globalParams.maxBatchSize);
    std::uniform_int_distribution<int> epoch_dist(globalParams.minEpochs, globalParams.maxEpochs);
    std::uniform_int_distribution<int> activation_dist(0, 3);  // fixed range for activation functions
    std::uniform_int_distribution<int> layer_dist(globalParams.minHiddenLayers, globalParams.maxHiddenLayers);
    std::uniform_int_distribution<int> size_dist(globalParams.minLayerSize, globalParams.maxLayerSize);

    // initialize basic hyperparameters
    genome.gene(0, lr_dist(gen));          // learning rate
    genome.gene(1, dropout_dist(gen));     // dropout rate
    genome.gene(2, static_cast<float>(batch_dist(gen)));    // batch size
    genome.gene(3, static_cast<float>(epoch_dist(gen)));    // epochs
    genome.gene(4, static_cast<float>(activation_dist(gen))); // activation function

    // randomly determine number of hidden layers
    int num_hidden_layers = layer_dist(gen);
    genome.gene(5, static_cast<float>(num_hidden_layers));

    // initialize hidden layer sizes
    // active layers get random sizes, unused layers are set to 0
    for (int i = 0; i < globalParams.maxHiddenLayers; ++i) {
        if (i < num_hidden_layers) {
            genome.gene(6 + i, static_cast<float>(size_dist(gen)));  // active layer
        } else {
            genome.gene(6 + i, 0.0f);  // inactive layer
        }
    }
}

/**
custom genome mutator for galib. applies random mutations to genes while ensuring values
stay within valid ranges. different mutation strategies are used for different parameter types:
- float parameters: small continuous changes
- integer parameters: small changes rounded to integers
- layer structure: maintains consistency between num_hidden_layers and layer sizes

@param g reference to the genome being mutated
@param mutationRate probability of each gene being mutated
@return always returns 1 to indicate mutation occurred
**/
int customMutator(GAGenome& g, float mutationRate) {
    auto& genome = dynamic_cast<GA1DArrayGenome<float>&>(g);

    // define valid ranges for each basic parameter
    const std::vector<std::pair<float, float>> validRanges = {
        {globalParams.minLearningRate, globalParams.maxLearningRate},     // learning rate
        {globalParams.minDropoutRate, globalParams.maxDropoutRate},       // dropout rate
        {static_cast<float>(globalParams.minBatchSize),
         static_cast<float>(globalParams.maxBatchSize)},                             // batch size
        {static_cast<float>(globalParams.minEpochs),
         static_cast<float>(globalParams.maxEpochs)},                                // epochs
        {0.0f, 3.0f},                                                           // activation function
        {static_cast<float>(globalParams.minHiddenLayers),
         static_cast<float>(globalParams.maxHiddenLayers)}                           // number of hidden layers
    };

    // mutate basic parameters (genes 0-5)
    for (int i = 0; i < validRanges.size(); ++i) {
        if (GARandomFloat(0.0f, 1.0f) < mutationRate) {
            // apply random mutation
            float mutationAmount = GARandomFloat(-0.1f, 0.1f);
            genome.gene(i, genome.gene(i) + mutationAmount);

            // clamp to valid range, with special handling for float vs int parameters
            if (i == 0 || i == 1) {  // learning rate and dropout rate (floats)
                genome.gene(i, std::clamp(genome.gene(i),
                    validRanges[i].first, validRanges[i].second));
            } else {  // integer parameters
                genome.gene(i, std::round(std::clamp(genome.gene(i),
                    validRanges[i].first, validRanges[i].second)));
            }

            if (debug_mode) {
                std::cout << "[evolution]: mutation occurred in gene " << i << std::endl;
            }
        }
    }

    // get current number of hidden layers after possible mutation
    int num_hidden_layers = static_cast<int>(std::round(std::clamp(genome.gene(5),
        validRanges[5].first, validRanges[5].second)));

    // mutate hidden layer sizes (genes 6+)
    for (int i = 0; i < globalParams.maxHiddenLayers; ++i) {
        int geneIndex = 6 + i;
        if (i < num_hidden_layers) {  // active layer
            if (geneIndex >= genome.size()) {
                // initialize new layer if needed
                genome.gene(geneIndex, GARandomFloat(
                    static_cast<float>(globalParams.minLayerSize),
                    static_cast<float>(globalParams.maxLayerSize)));
            } else if (GARandomFloat(0.0f, 1.0f) < mutationRate) {
                // mutate existing layer size
                float mutationAmount = GARandomFloat(-10.0f, 10.0f);
                genome.gene(geneIndex, genome.gene(geneIndex) + mutationAmount);
                genome.gene(geneIndex, std::round(std::clamp(genome.gene(geneIndex),
                    static_cast<float>(globalParams.minLayerSize),
                    static_cast<float>(globalParams.maxLayerSize))));
            }
        } else {  // inactive layer
            genome.gene(geneIndex, 0.0f);
        }
    }

    return 1;  // indicate mutation occurred
}

/**
calculates the average fitness across all individuals in a population.
used to track evolution progress and check convergence criteria.

@param population reference to the GAPopulation to analyze
@return float value representing the average fitness of all individuals
**/
float calculateAverageFitness(const GAPopulation& population) {
    float totalFitness = 0.0f;
    int populationSize = population.size();
    // sum up fitness scores of all individuals
    for (int i = 0; i < populationSize; ++i) {
        totalFitness += population.individual(i).score();
    }
    // return average fitness
    return totalFitness / static_cast<float>(populationSize);
}

/**
prints detailed help information about the program's usage,
explaining all available command-line parameters, their meanings,
and their default or allowed ranges.
**/
void printHelp(const char* programName) {
    std::cout << "Neural Network Hyperparameter Optimization\n"
              << "Usage: " << programName << " [options]\n\n"
              << "This program uses a Genetic Algorithm to optimize neural network hyperparameters.\n"
              << "It can be run without any parameters, using default settings.\n\n"
              << "Available Options:\n"
              << "  --help                 Display this help message\n\n"
              << "Genetic Algorithm Parameters:\n"
              << "  --population <size>    Population size for genetic algorithm (default: 15, range: 1-1000)\n"
              << "  --generations <count>  Maximum number of generations (default: 5, range: 1-1000)\n"
              << "  --mutation <prob>      Mutation probability (default: 0.2, range: 0.0-1.0)\n"
              << "  --crossover <prob>     Crossover probability (default: 0.7, range: 0.0-1.0)\n"
              << "  --fitness-threshold <threshold>  Early stopping fitness threshold (default: 0.92, range: 0.0-1.0)\n"
              << "  --consecutive-gen <count>  Consecutive generations to meet threshold for early stopping (default: 5, range: 1-1000)\n\n"
              << "Neural Network Hyperparameter Boundaries:\n"
              << "  --min-lr <rate>        Minimum learning rate (range: 0.0-1.0)\n"
              << "  --max-lr <rate>        Maximum learning rate (range: 0.0-1.0)\n"
              << "  --min-dropout <rate>   Minimum dropout rate (range: 0.0-1.0)\n"
              << "  --max-dropout <rate>   Maximum dropout rate (range: 0.0-1.0)\n"
              << "  --min-batch <size>     Minimum batch size (range: 1-1000)\n"
              << "  --max-batch <size>     Maximum batch size (range: 1-1000)\n"
              << "  --min-epochs <count>   Minimum training epochs (range: 1-1000)\n"
              << "  --max-epochs <count>   Maximum training epochs (range: 1-1000)\n"
              << "  --min-layers <count>   Minimum hidden layers (range: 1-1000)\n"
              << "  --max-layers <count>   Maximum hidden layers (range: 1-1000)\n"
              << "  --min-layer-size <size> Minimum hidden layer size (range: 1-1000)\n"
              << "  --max-layer-size <size> Maximum hidden layer size (range: 1-1000)\n\n"
              << "Example Usage:\n"
              << "  " << programName << "  # Run with default settings\n"
              << "  " << programName << " --population 30 --generations 10 --mutation 0.1\n"
              << "  " << programName << " --min-lr 0.001 --max-lr 0.01 --min-batch 32 --max-batch 128\n\n"
              << "The program uses a Genetic Algorithm to explore and find optimal hyperparameters\n"
              << "for a neural network by minimizing the objective function.\n";
}

/**
main function orchestrating the genetic algorithm for neural network hyperparameter optimization.
handles command-line argument parsing, ga configuration, evolution process, and result logging.
supports early stopping when convergence criteria are met.

@param argc number of command-line arguments
@param argv array of command-line argument strings
@return 0 for successful execution, 1 for errors
**/
int main(int argc, char* argv[]) {
    // check for help flag first
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            printHelp(argv[0]);
            return 0;
        }
    }
    // parse command-line arguments
for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    // ensure each argument has a corresponding value
    if (i + 1 >= argc) {
        std::cerr << "error: missing value for " << arg << std::endl;
        return 1;
    }

    // process each argument and set corresponding parameter
    bool success = true;
    if (arg == "--population") {
        success = setIntParameter(globalParams.populationSize, argv[++i], 1, 1000, "population size");
    } else if (arg == "--generations") {
        success = setIntParameter(globalParams.nGenerations, argv[++i], 1, 1000, "generations");
    } else if (arg == "--mutation") {
        success = setFloatParameter(globalParams.pMutation, argv[++i], 0.0f, 1.0f, "mutation probability");
    } else if (arg == "--crossover") {
        success = setFloatParameter(globalParams.pCrossover, argv[++i], 0.0f, 1.0f, "crossover probability");
    } else if (arg == "--fitness-threshold") {
        success = setFloatParameter(globalParams.stoppingThreshold, argv[++i], 0.0f, 1.0f, "fitness threshold");
    } else if (arg == "--consecutive-gen") {
        success = setIntParameter(globalParams.requiredConsecutiveGenerations, argv[++i], 1, 1000, "consecutive generations");
    } else if (arg == "--min-lr") {
        success = setFloatParameter(globalParams.minLearningRate, argv[++i], 0.0f, 1.0f, "minimum learning rate");
    } else if (arg == "--max-lr") {
        success = setFloatParameter(globalParams.maxLearningRate, argv[++i], 0.0f, 1.0f, "maximum learning rate");
    } else if (arg == "--min-dropout") {
        success = setFloatParameter(globalParams.minDropoutRate, argv[++i], 0.0f, 1.0f, "minimum dropout rate");
    } else if (arg == "--max-dropout") {
        success = setFloatParameter(globalParams.maxDropoutRate, argv[++i], 0.0f, 1.0f, "maximum dropout rate");
    } else if (arg == "--min-batch") {
        success = setIntParameter(globalParams.minBatchSize, argv[++i], 1, 1000, "minimum batch size");
    } else if (arg == "--max-batch") {
        success = setIntParameter(globalParams.maxBatchSize, argv[++i], 1, 1000, "maximum batch size");
    } else if (arg == "--min-epochs") {
        success = setIntParameter(globalParams.minEpochs, argv[++i], 1, 1000, "minimum epochs");
    } else if (arg == "--max-epochs") {
        success = setIntParameter(globalParams.maxEpochs, argv[++i], 1, 1000, "maximum epochs");
    } else if (arg == "--min-layers") {
        success = setIntParameter(globalParams.minHiddenLayers, argv[++i], 1, 1000, "minimum hidden layers");
    } else if (arg == "--max-layers") {
        success = setIntParameter(globalParams.maxHiddenLayers, argv[++i], 1, 1000, "maximum hidden layers");
    } else if (arg == "--min-layer-size") {
        success = setIntParameter(globalParams.minLayerSize, argv[++i], 1, 1000, "minimum layer size");
    } else if (arg == "--max-layer-size") {
        success = setIntParameter(globalParams.maxLayerSize, argv[++i], 1, 1000, "maximum layer size");
    } else {
        std::cerr << "unknown argument: " << arg << std::endl;
        success = false;
    }

    // if parameter setting failed, display complete usage information
    if (!success) {
        std::cerr << "usage: " << argv[0] << "\n"
                  << "  [--population <size>] [--generations <count>]\n"
                  << "  [--mutation <prob>] [--crossover <prob>]\n"
                  << "  [--fitness-threshold <threshold>] [--consecutive-gen <count>]\n"
                  << "  [--min-lr <rate>] [--max-lr <rate>]\n"
                  << "  [--min-dropout <rate>] [--max-dropout <rate>]\n"
                  << "  [--min-batch <size>] [--max-batch <size>]\n"
                  << "  [--min-epochs <count>] [--max-epochs <count>]\n"
                  << "  [--min-layers <count>] [--max-layers <count>]\n"
                  << "  [--min-layer-size <size>] [--max-layer-size <size>]\n";
        return 1;
    }
}

    // validate relationships between min/max parameters
    if (globalParams.minLearningRate >= globalParams.maxLearningRate) {
        std::cerr << "error: minimum learning rate must be less than maximum learning rate" << std::endl;
        return 1;
    }
    if (globalParams.minDropoutRate >= globalParams.maxDropoutRate) {
        std::cerr << "error: minimum dropout rate must be less than maximum dropout rate" << std::endl;
        return 1;
    }
    if (globalParams.minBatchSize >= globalParams.maxBatchSize) {
        std::cerr << "error: minimum batch size must be less than maximum batch size" << std::endl;
        return 1;
    }
    if (globalParams.minEpochs >= globalParams.maxEpochs) {
        std::cerr << "error: minimum epochs must be less than maximum epochs" << std::endl;
        return 1;
    }
    if (globalParams.minHiddenLayers >= globalParams.maxHiddenLayers) {
        std::cerr << "error: minimum hidden layers must be less than maximum hidden layers" << std::endl;
        return 1;
    }
    if (globalParams.minLayerSize >= globalParams.maxLayerSize) {
        std::cerr << "error: minimum layer size must be less than maximum layer size" << std::endl;
        return 1;
    }

    // print configured parameters
    std::cout << "-----------------------------------------------------------------------------------------" << std::endl;
    std::cout << "[config]: population size: " << globalParams.populationSize << std::endl;
    std::cout << "[config]: number of generations: " << globalParams.nGenerations << std::endl;
    std::cout << "[config]: mutation probability: " << globalParams.pMutation << std::endl;
    std::cout << "[config]: crossover probability: " << globalParams.pCrossover << std::endl;
    std::cout << "-----------------------------------------------------------------------------------------" << std::endl;

    // initialize logging
    logFile.open(getLogFilePath(), std::ios::out);
    if (logFile.is_open()) {
        logFile << "LearningRate,DropoutRate,BatchSize,Epochs,ActivationFunction,NumHiddenLayers,HiddenSizes,Fitness\n";
        std::cout << "-----------------------------------------------------------------------------------------" << std::endl;
        std::cout << "[logfile]: " << getLogFilePath() << std::endl;
        std::cout << "-----------------------------------------------------------------------------------------" << std::endl;
        logFile.flush();
    } else {
        std::cerr << "[error]: unable to open log file!" << std::endl;
        return 1;
    }

    // configure genetic algorithm
    const int max_hidden_layers = globalParams.maxHiddenLayers;  // maximum number of hidden layers supported
    const int genome_size = 6 + max_hidden_layers;  // genes for params + layer sizes

    // create and configure genome template
    GA1DArrayGenome<float> genome(genome_size, Objective);
    genome.initializer(customInitializer);
    genome.mutator(customMutator);

    // create and configure ga instance
    GASimpleGA ga(genome);
    ga.populationSize(globalParams.populationSize);
    ga.nGenerations(globalParams.nGenerations);
    ga.pMutation(globalParams.pMutation);
    ga.pCrossover(globalParams.pCrossover);
    ga.elitist(static_cast<GABoolean>(1));  // enable elitism to preserve best solutions

    // configure scaling and termination
    GANoScaling scaling;
    ga.scaling(scaling);
    ga.maximize();  // we want to maximize fitness
    ga.terminator(GAGeneticAlgorithm::TerminateUponGeneration);
    ga.scoreFrequency(1);   // evaluate fitness every generation
    ga.flushFrequency(1);   // update statistics every generation

    // initialize evolution process
    if (debug_mode) {
        std::cout << "-----------------------------------------------------------------------------------------" << std::endl;
    }
    ga.initialize();

    // evaluate initial population
    float avgFitness = calculateAverageFitness(ga.population());
    if (debug_mode) {
        std::cout << "-----------------------------------------------------------------------------------------" << std::endl;
        std::cout << "[generation 0] average fitness: " << avgFitness << std::endl;
        std::cout << "-----------------------------------------------------------------------------------------" << std::endl;
    }

    // log initial generation results
    if (logFile.is_open()) {
        logFile << "[generation 0] average fitness: " << avgFitness << "\n";
        logFile.flush();
    }

    // evolution loop
    int consecutiveGenerations = 0;  // counter for convergence check

    for (int generation = 1; generation < globalParams.nGenerations; ++generation) {
        // verify population size
        if (ga.population().size() != globalParams.populationSize) {
            std::cerr << "[error]: Population size mismatch in generation " << generation
                      << ". Expected: " << globalParams.populationSize
                      << ", Actual: " << ga.population().size() << std::endl;
        }

        // evolve population by one generation
        ga.step();

        // calculate and log fitness metrics
        avgFitness = calculateAverageFitness(ga.population());
        if (debug_mode) {
            std::cout << "-----------------------------------------------------------------------------------------" << std::endl;
            std::cout << "[generation " << generation << "] average fitness: " << avgFitness << std::endl;
            std::cout << "-----------------------------------------------------------------------------------------" << std::endl;
        }
        if (logFile.is_open()) {
            logFile << "[generation " << generation << "] average fitness: " << avgFitness << "\n";
            logFile.flush();
        }

        // check for convergence (early stopping)
        if (avgFitness >= globalParams.stoppingThreshold) {
            ++consecutiveGenerations;
            if (consecutiveGenerations >= globalParams.requiredConsecutiveGenerations) {
                std::cout << "-----------------------------------------------------------------------------------------" << std::endl;
                std::cout << "[stopping]: average fitness ≥ " << globalParams.stoppingThreshold
                          << " for " << globalParams.requiredConsecutiveGenerations
                          << " consecutive generations." << std::endl;
                std::cout << "-----------------------------------------------------------------------------------------" << std::endl;
                break;
            }
        } else {
            consecutiveGenerations = 0;  // reset counter if threshold not met
        }
    }

    // output best solution found
    const auto& bestGenome = dynamic_cast<const GA1DArrayGenome<float>&>(ga.statistics().bestIndividual());
    std::cout << "-----------------------------------------------------------------------------------------" << std::endl;
    std::cout << "[best]: learning rate: " << bestGenome.gene(0) << std::endl;
    std::cout << "[best]: dropout rate: " << bestGenome.gene(1) << std::endl;
    std::cout << "[best]: batch size: " << static_cast<int>(bestGenome.gene(2)) << std::endl;
    std::cout << "[best]: epochs: " << static_cast<int>(bestGenome.gene(3)) << std::endl;
    std::cout << "[best]: activation function: " << static_cast<int>(bestGenome.gene(4)) << std::endl;
    std::cout << "[best]: number of hidden layers: " << static_cast<int>(bestGenome.gene(5)) << std::endl;

    // output hidden layer configuration
    std::cout << "[best]: hidden layer sizes: [";
    int num_hidden_layers = static_cast<int>(bestGenome.gene(5));
    for (int i = 0; i < num_hidden_layers; ++i) {
        if (i > 0) std::cout << ", ";
        std::cout << static_cast<int>(bestGenome.gene(6 + i));
    }
    std::cout << "]" << std::endl;

    std::cout << "[best]: fitness: " << bestGenome.score() << std::endl;
    std::cout << "-----------------------------------------------------------------------------------------" << std::endl;

    // cleanup
    if (logFile.is_open()) {
        logFile.close();
    }

    return 0;
}
