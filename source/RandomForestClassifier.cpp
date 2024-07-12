
#include "RandomForestClassifier.h"

#include <fstream>
#include <iostream>
#include <algorithm>
#include <sstream>
#include <memory> // Include TR1 shared_ptr

RandomForestClassifier::RandomForestClassifier(BaseVectorizer* pvec)
    : num_trees(50), max_depth(5) // Initialize with default values
{
    pVec = pvec;
}

RandomForestClassifier::~RandomForestClassifier()
{
    delete pVec;
}

void RandomForestClassifier::setHyperparameters(std::string hyperparameters)
{
    std::string token;
    std::istringstream tokenStream(hyperparameters);

    // "num_trees=50,max_depth=5"
    num_trees = 50;
    max_depth = 5;

    while (std::getline(tokenStream, token, ','))
    {
        std::istringstream pairStream(token);
        std::string key;
        double value;

        if (std::getline(pairStream, key, '=') && pairStream >> value)
        {
            std::cout << key << " = " << value << std::endl;
            if (key == "minfrequency")
            {
                minfrequency = value;
            }
            else if (key == "num_trees")
            {
                num_trees = value;
            }
            else if (key == "max_depth")
            {
                max_depth = value;
            }
        }
    }
}

void RandomForestClassifier::fit(std::string abs_filepath_to_features, std::string abs_filepath_to_labels)
{
    if (minfrequency > 0)
    {
        pVec->scanForSparseHistogram(abs_filepath_to_features, minfrequency);
    }
    pVec->fit(abs_filepath_to_features, abs_filepath_to_labels);
    std::vector<std::tr1::shared_ptr<Sentence> > sentences = pVec->sentences;

    for (int i = 0; i < num_trees; ++i)
    {
        std::tr1::shared_ptr<DecisionTree> tree(new DecisionTree(max_depth));
        tree->fit(sentences);
        trees.push_back(tree);
    }
}

Prediction RandomForestClassifier::predict(std::string sentence, bool preprocess)
{
    GlobalData vars;
    Prediction result;

    std::vector<std::string> processed_input = pVec->buildSentenceVector(sentence, preprocess);
    std::vector<double> feature_vector = pVec->getSentenceFeatures(processed_input);

    std::vector<int> votes(3, 0); // Assuming 3 classes: POS, NEG, NEU
    for (std::vector<std::tr1::shared_ptr<DecisionTree> >::const_iterator it = trees.begin(); it != trees.end(); ++it)
    {
        int prediction = (*it)->predict(feature_vector).label;
        votes[prediction]++;
    }

    std::vector<double> probabilities(3, 0.0);
    for (size_t i = 0; i < votes.size(); ++i)
    {
        probabilities[i] = static_cast<double>(votes[i]) / trees.size();
    }

    int max_index = std::distance(votes.begin(), std::max_element(votes.begin(), votes.end()));

    result.probability = probabilities[1];

    switch (max_index)
    {
    case 0:
        result.label = vars.NEG;
        break;
    case 1:
        result.label = vars.POS;
        break;
    case 2:
        result.label = vars.NEU;
        break;
    default:
        break;
    }

    return result;
}

void RandomForestClassifier::predict(std::string abs_filepath_to_features, std::string abs_filepath_to_labels, bool preprocess)
{
    std::ifstream in(abs_filepath_to_features.c_str());
    std::ofstream out(abs_filepath_to_labels.c_str());
    std::string feature_input;

    if (!in)
    {
        std::cerr << "ERROR: Cannot open features file.\n";
        return;
    }

    if (!out)
    {
        std::cerr << "ERROR: Cannot open labels file.\n";
        return;
    }

#ifdef BENCHMARK
    double sumduration = 0.0;
    double sumstrlen = 0.0;
    size_t num_rows = 0;
#endif

    while (getline(in, feature_input))
    {
#ifdef BENCHMARK
        clock_t start = clock();
#endif

        Prediction result = predict(feature_input, preprocess);

#ifdef BENCHMARK
        clock_t end = clock();
        double milliseconds = static_cast<double>(end - start) / CLOCKS_PER_SEC * 1000.0;
        sumduration += milliseconds;
        sumstrlen += feature_input.length();
        num_rows++;
#endif

        out << result.label << "," << result.probability << std::endl;
    }

#ifdef BENCHMARK
    double avgduration = sumduration / num_rows;
    std::cout << "Average Time per Text = " << avgduration << " ms" << std::endl;
    double avgstrlen = sumstrlen / num_rows;
    std::cout << "Average Length of Text (chars) = " << avgstrlen << std::endl;
#endif

    in.close();
    out.close();
}

void RandomForestClassifier::save(const std::string& filename) const
{
    std::ofstream outFile(filename.c_str(), std::ios::binary);
    if (!outFile.is_open())
    {
        std::cerr << "Failed to open file for writing." << std::endl;
        return;
    }

    pVec->save(outFile);

    size_t num_trees = trees.size();
    outFile.write(reinterpret_cast<const char*>(&num_trees), sizeof(num_trees));
    for (std::vector<std::tr1::shared_ptr<DecisionTree> >::const_iterator it = trees.begin(); it != trees.end(); ++it)
    {
        (*it)->save(outFile);
    }

    outFile.close();
}

void RandomForestClassifier::load(const std::string& filename)
{
    std::ifstream inFile(filename.c_str(), std::ios::binary);
    if (!inFile.is_open())
    {
        std::cerr << "Failed to open file for reading." << std::endl;
        return;
    }

    pVec->load(inFile);

    size_t num_trees;
    inFile.read(reinterpret_cast<char*>(&num_trees), sizeof(num_trees));
    trees.resize(num_trees);
    for (size_t i = 0; i < num_trees; ++i)
    {
        std::tr1::shared_ptr<DecisionTree> tree(new DecisionTree());
        tree->load(inFile);
        trees[i] = tree;
    }

    inFile.close();
}
