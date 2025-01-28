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

bool debug_mode = true; // flag for printing debug messages
std::ofstream logFile;  // global log file stream


/*
generates a unique log file path by creating a "logs" directory (if it doesn't exist)
and appending a timestamped filename.

@return a std::string representing the full path to the log file.
        the file path includes the "logs" directory and a filename
        in the format "evolution_log_<yyyy-mm-dd_hh-mm-ss>.csv".
*/
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


/*
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
@return a float value representing the fitness score of the neural network
*/
float evaluate_nn(const std::vector<float>& params) {
    float learning_rate = params[0];
    float dropout_rate = params[1];
    int batch_size = static_cast<int>(params[2]);
    int epochs = static_cast<int>(params[3]);
    int activation_function = static_cast<int>(params[4]);
    int num_hidden_layers = static_cast<int>(params[5]);

    // dynamically build the hidden_sizes array based on num_hidden_layers
    std::ostringstream hidden_sizes;
    hidden_sizes << "[";
    for (int i = 0; i < num_hidden_layers; ++i) {
        if (static_cast<int>(params[6 + i]) <= 0) {
            return 0.0f; // if the condition is not met, gene mutated to invalid combination
        }
        if (i > 0) {
            hidden_sizes << ",";
        }
        hidden_sizes << static_cast<int>(params[6 + i]); // append the hidden layer size
    }
    hidden_sizes << "]";


    // construct the python command
    std::ostringstream cmd;
    cmd << "python ../evol_neuralnet.py \""
        << R"({\"learning_rate\":)" << learning_rate
        << R"(, \"dropout_rate\":)" << dropout_rate
        << R"(, \"batch_size\":)" << batch_size
        << R"(, \"epochs\":)" << epochs
        << R"(, \"activation_function\":)" << activation_function
        << R"(, \"num_hidden_layers\":)" << num_hidden_layers
        << R"(, \"hidden_sizes\":)" << hidden_sizes.str()
        << "}\"";

    if (debug_mode) {
        std::cout << "[training]: executing command: " << cmd.str() << std::endl;
    }

    FILE* pipe = _popen((cmd.str() + " 2>&1").c_str(), "r");  // redirect stderr to stdout
    if (!pipe) {
        std::cerr << "error: could not open pipe to python script!" << std::endl;
        float fitness = 0.0f;
        if (logFile.is_open()) {
            logFile << std::fixed << std::setprecision(8)
            << params[0] << "," << params[1] << "," << batch_size << ","
            << epochs << "," << activation_function << "," << num_hidden_layers << ",\"["; // Start of the list with quotes
            for (int i = 0; i < num_hidden_layers; ++i) {
                if (i > 0) logFile << ", ";
                logFile << static_cast<int>(params[6 + i]);
            }
            logFile << "]\"" << "," << fitness << "\n"; // End of the list with quotes
            logFile.flush();
        }
        return fitness;  // default fitness
    }

    // Read the final output only
    char buffer[128];
    std::string result;
    if (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
        result = buffer;  // read just the first line of output (fitness score)
    }

    // Close the process and get the return code
    int returnCode = _pclose(pipe);
    if (returnCode != 0) {
        std::cerr << "error: python script failed with return code " << returnCode << std::endl;
        float fitness = 0.0f;
        if (logFile.is_open()) {
            logFile << std::fixed << std::setprecision(8)
            << params[0] << "," << params[1] << "," << batch_size << ","
            << epochs << "," << activation_function << "," << num_hidden_layers << ",\"["; // Start of the list with quotes
            for (int i = 0; i < num_hidden_layers; ++i) {
                if (i > 0) logFile << ", ";
                logFile << static_cast<int>(params[6 + i]);
            }
            logFile << "]\"" << "," << fitness << "\n"; // End of the list with quotes
            logFile.flush();
            }
        return fitness;  // default fitness
    }

    // parse the output as a float representing fitness
    try {
        float fitness = std::stof(result);
        if (debug_mode) {
            std::cout << "[training]: fitness score: " << result << std::endl;
        }
        if (logFile.is_open()) {
            logFile << std::fixed << std::setprecision(8)
            << params[0] << "," << params[1] << "," << batch_size << ","
            << epochs << "," << activation_function << "," << num_hidden_layers << ",\"["; // Start of the list with quotes
            for (int i = 0; i < num_hidden_layers; ++i) {
                if (i > 0) logFile << ", ";
                    logFile << static_cast<int>(params[6 + i]);
            }
            logFile << "]\"" << "," << fitness << "\n"; // End of the list with quotes
            logFile.flush();
        }
        return fitness;
    } catch (const std::exception& e) {
        std::cerr << "[error]: invalid output from python script: " << e.what() << std::endl;
        return 0.0f;  // default fitness
    }
}


/*
objective function for galib. evaluates the fitness of a genome
by passing its parameters to the evaluate_nn function.
@param g a reference to the genome being evaluated
@return a float value representing the fitness score of the genome
*/
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


/*
custom initializer for the genome. randomly initializes hyperparameters
such as learning rate, dropout rate, batch size, epochs, and hidden layer sizes.
@param g a reference to the genome being initialized
*/
void customInitializer(GAGenome& g) {
    auto& genome = dynamic_cast<GA1DArrayGenome<float>&>(g);

    std::random_device rd;
    std::mt19937 gen(rd());

    std::uniform_real_distribution<float> lr_dist(0.0001f, 0.1f);   // learning rate
    std::uniform_real_distribution<float> dropout_dist(0.0f, 0.5f); // dropout rate
    std::uniform_int_distribution<int> batch_dist(16, 128);         // batch size
    std::uniform_int_distribution<int> epoch_dist(1, 50);           // epochs
    std::uniform_int_distribution<int> activation_dist(0, 3);       // activation function
    std::uniform_int_distribution<int> layer_dist(1, 5);            // number of hidden layers
    std::uniform_int_distribution<int> size_dist(16, 128);          // hidden layer sizes

    genome.gene(0, lr_dist(gen));                              // learning rate
    genome.gene(1, dropout_dist(gen));                         // dropout rate
    genome.gene(2, static_cast<float>(batch_dist(gen)));       // batch size
    genome.gene(3, static_cast<float>(epoch_dist(gen)));       // epochs
    genome.gene(4, static_cast<float>(activation_dist(gen)));  // activation function
    int num_hidden_layers = layer_dist(gen);
    genome.gene(5, static_cast<float>(num_hidden_layers));     // number of hidden layers

    // initialize sizes for all possible hidden layers
    int max_hidden_layers = 5; // maximum number of hidden layers
    for (int i = 0; i < max_hidden_layers; ++i) {
        if (i < num_hidden_layers) {
            genome.gene(6 + i, static_cast<float>(size_dist(gen))); // active hidden layer size
        } else {
            genome.gene(6 + i, 0.0f); // explicitly set unused layers to 0
        }
    }
}


/*
custom mutator for the genome.
modifies genome values by applying random mutations to parameters and ensures
all values remain within valid ranges by clamping them.
@param g a reference to the genome being mutated
@param mutationRate the probability of mutating a specific gene
@return 1 to indicate that mutation occurred
*/
int customMutator(GAGenome& g, const float mutationRate) {
    auto& genome = dynamic_cast<GA1DArrayGenome<float>&>(g);

    // define valid ranges for parameters
    const std::vector<std::pair<float, float>> validRanges = {
        {0.0001f, 0.1f},  // learning rate (float)
        {0.0f, 0.5f},     // dropout rate (float)
        {16.0f, 128.0f},  // batch size (int)
        {1.0f, 50.0f},    // epochs (int)
        {0.0f, 3.0f},     // activation function (int)
        {1.0f, 5.0f}      // number of hidden layers (int)
    };

    const int maxHiddenLayers = static_cast<int>(validRanges[5].second);

    // mutate standard parameters (indices 0–5)
    for (int i = 0; i < validRanges.size(); ++i) {
        if (GARandomFloat(0.0f, 1.0f) < mutationRate) {  // check if this gene should mutate
            float mutationAmount = GARandomFloat(-0.1f, 0.1f);  // mutation step
            genome.gene(i, genome.gene(i) + mutationAmount);

            // clamp to valid range and round where needed
            if (i == 0 || i == 1) {  // learning rate and dropout rate (floats)
                genome.gene(i, std::clamp(genome.gene(i), validRanges[i].first, validRanges[i].second));
            } else {  // batch size, epochs, activation function, and hidden layers (integers)
                genome.gene(i, std::round(std::clamp(genome.gene(i), validRanges[i].first, validRanges[i].second)));
            }
            if (debug_mode) {
                std::cout << "[evolution]: mutation occurred in gene " << i << std::endl;
            }
        }
    }

    // get updated number of hidden layers
    int num_hidden_layers = static_cast<int>(std::round(std::clamp(genome.gene(5), validRanges[5].first, validRanges[5].second)));

    // apply mutation to hidden layer sizes (indices 6 to 6 + maxHiddenLayers - 1)
    for (int i = 0; i < maxHiddenLayers; ++i) {
        int geneIndex = 6 + i;  // index of the hidden layer size
        if (i < num_hidden_layers) {  // active layers
            if (geneIndex >= genome.size()) {
                // if this is a new layer, initialize it
                genome.gene(geneIndex, GARandomFloat(16.0f, 128.0f));  // assign random valid size
            } else if (GARandomFloat(0.0f, 1.0f) < mutationRate) {
                // mutate existing active layers
                float mutationAmount = GARandomFloat(-10.0f, 10.0f);  // mutation step
                genome.gene(geneIndex, genome.gene(geneIndex) + mutationAmount);
                genome.gene(geneIndex, std::round(std::clamp(genome.gene(geneIndex), 16.0f, 128.0f)));
            }
        } else {
            // Reset unused hidden layer sizes
            genome.gene(geneIndex, 0.0f);
        }
    }

    return 1;  // indicates mutation occurred
}


float calculateAverageFitness(const GAPopulation& population) {
    float totalFitness = 0.0f;
    int populationSize = population.size();
    for (int i = 0; i < populationSize; ++i) {
        totalFitness += population.individual(i).score();
    }
    return totalFitness / static_cast<float>(populationSize);
}


/*
main function. sets up and configures the genetic algorithm,
then evolves the population to find optimal hyperparameters
for the neural network.
@param argc the number of command-line arguments
@param argv the command-line arguments
@return an integer representing the exit code (0 for success)
*/
int main(int argc, char* argv[]) {
    // default ga parameters
    int populationSize = 10;       // population size
    int nGenerations = 15;         // number of generations
    float pMutation = 0.2f;        // mutation probability
    float pCrossover = 0.7f;       // crossover probability
    float stoppingThreshold = 0.92f; // fitness threshold for stopping
    int requiredConsecutiveGenerations = 5; // required consecutive generations
    int consecutiveGenerations = 0; // counter for consecutive generations

    // parse command-line arguments
    for (int i = 1; i < argc; ++i) {
        if (std::string arg = argv[i]; arg == "--population" && i + 1 < argc) {
            populationSize = std::stoi(argv[++i]);
        } else if (arg == "--generations" && i + 1 < argc) {
            nGenerations = std::stoi(argv[++i]);
        } else if (arg == "--mutation" && i + 1 < argc) {
            pMutation = std::stof(argv[++i]);
        } else if (arg == "--crossover" && i + 1 < argc) {
            pCrossover = std::stof(argv[++i]);
        } else {
            std::cerr << "unknown argument: " << arg << std::endl;
            std::cerr << "usage: " << argv[0]
                      << " [--population <size>] [--generations <count>] "
                      << "[--mutation <probability>] [--crossover <probability>]" << std::endl;
            return 1;
        }
    }

    // print configured parameters
    std::cout << "[config]: population size: " << populationSize << std::endl;
    std::cout << "[config]: number of generations: " << nGenerations << std::endl;
    std::cout << "[config]: mutation probability: " << pMutation << std::endl;
    std::cout << "[config]: crossover probability: " << pCrossover << std::endl;

    // open the log file for writing
    logFile.open(getLogFilePath(), std::ios::out);
    if (logFile.is_open()) {
        logFile << "LearningRate,DropoutRate,BatchSize,Epochs,ActivationFunction,NumHiddenLayers,HiddenSizes,Fitness\n";
        std::cout << "[logfile]: " << getLogFilePath() << std::endl;
        logFile.flush();
    } else {
        std::cerr << "[error]: unable to open log file!" << std::endl;
        return 1; // exit if logging fails
    }

    // create and configure the genetic algorithm
    constexpr int max_hidden_layers = 5;
    constexpr int genome_size = 6 + max_hidden_layers;
    GA1DArrayGenome<float> genome(genome_size, Objective);
    genome.initializer(customInitializer);
    genome.mutator(customMutator);
    std::cout << "[debug]: setting ga parameters" << std::endl;
    GASimpleGA ga(genome);
    ga.populationSize(populationSize);
    ga.nGenerations(nGenerations);
    ga.pMutation(pMutation);
    ga.pCrossover(pCrossover);
    ga.elitist(static_cast<GABoolean>(0));

    std::cout << "[debug]: setting ga scaling" << std::endl;
    GANoScaling scaling;
    ga.scaling(scaling);

    // Add these lines to force evaluation in each generation
    ga.minimize();  // We're maximizing fitness
    ga.terminator(GAGeneticAlgorithm::TerminateUponGeneration);
    ga.scoreFrequency(1);   // Score every generation
    ga.flushFrequency(1);   // Flush scores every generation

    // evolve the population
    // runs through all generations (nGenerations)
    // ga.evolve();

    // stepwise evolution
    // handles early termination to avoid overfitting
    std::cout << "[debug]: initialize ga" << std::endl;
    ga.initialize();
    std::cout << "[debug]: population after initialization " << ga.population().size() << std::endl;
    std::cout << "[debug]: enter evolution loop" << std::endl;
    for (int generation = 0; generation < nGenerations; ++generation) {
        // validate population size
        std::cout << "[debug]: population " << ga.population().size() << std::endl;
        std::cout << "[debug]: generation " << generation << std::endl;
        if (ga.population().size() != populationSize) {
            std::cerr << "[error]: Population size mismatch in generation " << generation
                      << ". Expected: " << populationSize
                      << ", Actual: " << ga.population().size() << std::endl;
        }
        std::cout << "[debug]: starting step for  " << generation << std::endl;

        std::cout << "[debug]: pre-step individuals:\n";
        for (int i = 0; i < ga.population().size(); i++) {
            const auto& genome = dynamic_cast<const GA1DArrayGenome<float>&>(ga.population().individual(i));
            std::cout << "  Individual " << i << ": score=" << genome.score() << "\n";
        }

        ga.step();

        std::cout << "[debug]: post-step population size: " << ga.population().size() << std::endl;

        // Print details of each individual after step
        std::cout << "[debug]: post-step individuals:\n";
        for (int i = 0; i < ga.population().size(); i++) {
            const auto& genome = dynamic_cast<const GA1DArrayGenome<float>&>(ga.population().individual(i));
            std::cout << "  Individual " << i << ": score=" << genome.score() << "\n";
        }

        std::cout << "[debug]: finished step for  " << generation << std::endl;

        // Calculate average fitness using existing scores
        std::cout << "[debug]: calculating fitness for  " << generation << std::endl;
        float avgFitness = calculateAverageFitness(ga.population());

        if (debug_mode) {
            std::cout << "[generation " << generation << "]: average fitness: " << avgFitness << std::endl;
        }
        std::cout << "[debug]: start logging fitness for  " << generation << std::endl;
        // Log the results
        if (logFile.is_open()) {
            logFile << "[generation " << generation << "] average fitness: " << avgFitness << "\n";
            logFile.flush();
        }
        std::cout << "[debug]: finished logging fitness for  " << generation << std::endl;

        std::cout << "[debug]: check threshold fitness after  " << generation << std::endl;
        // Early stopping check
        if (avgFitness >= stoppingThreshold) {
            ++consecutiveGenerations;
            if (consecutiveGenerations >= requiredConsecutiveGenerations) {
                std::cout << "[stopping]: average fitness ≥ " << stoppingThreshold
                          << " for " << requiredConsecutiveGenerations << " consecutive generations." << std::endl;
                if (logFile.is_open()) {
                    logFile << "[stopping]: average fitness ≥ " << stoppingThreshold
                          << " for " << requiredConsecutiveGenerations << " consecutive generations.\n";
                    logFile.flush();
                } else {
                    std::cerr << "[error]: unable to open log file!" << std::endl;
                    return 1; // exit if logging fails
                }
                break;
            }
        } else {
            std::cout << "[debug]: threshold not met after  " << generation << std::endl;
            consecutiveGenerations = 0;
        }
    }

    // output the best result
    const auto& bestGenome = dynamic_cast<const GA1DArrayGenome<float>&>(ga.statistics().bestIndividual());

    std::cout << "[best]: learning rate: " << bestGenome.gene(0) << std::endl;
    std::cout << "[best]: dropout rate: " << bestGenome.gene(1) << std::endl;
    std::cout << "[best]: batch size: " << static_cast<int>(bestGenome.gene(2)) << std::endl;
    std::cout << "[best]: epochs: " << static_cast<int>(bestGenome.gene(3)) << std::endl;
    std::cout << "[best]: activation function: " << static_cast<int>(bestGenome.gene(4)) << std::endl;
    std::cout << "[best]: number of hidden layers: " << static_cast<int>(bestGenome.gene(5)) << std::endl;

    std::cout << "[best]: hidden layer sizes: [";
    int num_hidden_layers = static_cast<int>(bestGenome.gene(5));
    for (int i = 0; i < num_hidden_layers; ++i) {
        if (i > 0) std::cout << ", ";
        std::cout << static_cast<int>(bestGenome.gene(6 + i));
    }
    std::cout << "]" << std::endl;

    std::cout << "[best]: fitness: " << bestGenome.score() << std::endl;

    if (logFile.is_open()) {
        logFile.close();
    }

    return 0;
}
