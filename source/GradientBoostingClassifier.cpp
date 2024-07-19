
/*++

Revision History:
	Date:	Jun 28, 2024.
	Author:	Rajas Chavadekar.
	Desc:	Created.

--*/

#include "GradientBoostingClassifier.h"

#include <fstream>
#include <iostream>
#include <cmath>
#include <memory> // Include for std::tr1::shared_ptr
#include <unordered_map> // Include for std::tr1::unordered_map

GradientBoostingClassifier::GradientBoostingClassifier(BaseVectorizer* pvec)
    : n_trees(50), max_depth(5), learning_rate(0.01)
{
	pVec = pvec;
}

GradientBoostingClassifier::~GradientBoostingClassifier()
{
    delete pVec;
}

void GradientBoostingClassifier::setHyperparameters(std::string hyperparameters)
{
    std::string token;
    std::istringstream tokenStream(hyperparameters);

    // "n_trees=50,max_depth=5,learning_rate=0.01"
    n_trees = 50;
    max_depth = 5;
    learning_rate = 0.01;

    while (std::getline(tokenStream, token, ',')) {
        std::istringstream pairStream(token);
        std::string key;
        double value;

        if (std::getline(pairStream, key, '=') && pairStream >> value) {
            std::cout << key << " = " << value << std::endl;
            if (key == "minfrequency") {
                minfrequency = value;
            }
            if (key == "n_trees") {
                n_trees = value;
            }
            else if (key == "max_depth") {
                max_depth = value;
            }
            else if (key == "learning_rate") {
                learning_rate = value;
            }
        }
    }
}

void GradientBoostingClassifier::fit(std::string abs_filepath_to_features, std::string abs_filepath_to_labels)
{
    if (minfrequency > 0)
    {
        pVec->scanForSparseHistogram(abs_filepath_to_features, minfrequency);
    }
    pVec->fit(abs_filepath_to_features, abs_filepath_to_labels);

    size_t num_features = pVec->word_array.size();
    trees.clear();

    std::vector<std::tr1::shared_ptr<Sentence>> sentences = pVec->sentences;
    std::vector<double> labels(sentences.size());

    std::ifstream label_file(abs_filepath_to_labels.c_str());
    std::string label;
    for (size_t i = 0; i < labels.size(); ++i)
    {
        label_file >> labels[i];
    }
    label_file.close();

    std::vector<double> residuals(labels.size());

    for (int i = 0; i < n_trees; ++i)
    {
        for (size_t j = 0; j < sentences.size(); ++j)
        {
            std::vector<double> features;
            const std::tr1::unordered_map<int, double>& sentence_map = sentences[j]->sentence_map;
            features = pVec->getFrequencies(sentence_map);
            double y_true = labels[j];
            double y_pred = predict_proba(features);
            residuals[j] = y_true - y_pred;
        }

        std::tr1::shared_ptr<DecisionTree> tree(new DecisionTree(max_depth));
        tree->fit(sentences);
        trees.push_back(tree);
    }
}

Prediction GradientBoostingClassifier::predict(std::string sentence, bool preprocess)
{
    GlobalData vars;
    Prediction result;
    std::vector<std::string> processed_input = pVec->buildSentenceVector(sentence, preprocess);
    std::vector<double> feature_vector = pVec->getSentenceFeatures(processed_input);
    double probability = predict_proba(feature_vector);

    result.probability = probability;

    if (probability > 0.5)
    {
        result.label = vars.POS;
    }
    else
    {
        result.label = vars.NEG;
    }

    return result;
}

void GradientBoostingClassifier::predict(std::string abs_filepath_to_features, std::string abs_filepath_to_labels, bool preprocess)
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
    clock_t start;
    double sumduration = 0.0;
    double sumstrlen = 0.0;
    size_t num_rows = 0;
    #endif

    while (getline(in, feature_input))
    {
        #ifdef BENCHMARK
        start = clock();
        #endif

        Prediction result = predict(feature_input, preprocess);

        #ifdef BENCHMARK
        sumduration += static_cast<double>(clock() - start) / CLOCKS_PER_SEC * 1000.0;
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

double GradientBoostingClassifier::predict_tree(const DecisionTree& tree, const std::vector<double>& features) const
{
    return tree.predict(features).label;
}

double GradientBoostingClassifier::predict_proba(const std::vector<double>& features) const
{
    double score = 0.0;
    for (size_t i = 0; i < trees.size(); ++i)
    {
        double tree_prediction = predict_tree(*trees[i], features);
        score += learning_rate * tree_prediction;
    }
    return 1.0 / (1.0 + exp(-score));
}

void GradientBoostingClassifier::save(const std::string& filename) const
{
    std::ofstream outFile(filename.c_str(), std::ios::binary);
    if (!outFile.is_open())
    {
        std::cerr << "Failed to open file for writing." << std::endl;
        return;
    }

    pVec->save(outFile);

    size_t tree_count = trees.size();
    outFile.write(reinterpret_cast<const char*>(&tree_count), sizeof(tree_count));

    for (size_t i = 0; i < tree_count; ++i)
    {
        trees[i]->save(outFile);
    }

    outFile.write(reinterpret_cast<const char*>(&n_trees), sizeof(n_trees));
    outFile.write(reinterpret_cast<const char*>(&max_depth), sizeof(max_depth));
    outFile.write(reinterpret_cast<const char*>(&learning_rate), sizeof(learning_rate));
    
    outFile.close();
}

void GradientBoostingClassifier::load(const std::string& filename)
{
    std::ifstream inFile(filename.c_str(), std::ios::binary);
    if (!inFile.is_open())
    {
        std::cerr << "Failed to open file for reading." << std::endl;
        return;
    }

    pVec->load(inFile);

    size_t tree_count;
    inFile.read(reinterpret_cast<char*>(&tree_count), sizeof(tree_count));
    trees.resize(tree_count);
    for (size_t i = 0; i < tree_count; ++i)
    {
        trees[i].reset(new DecisionTree(max_depth));
        trees[i]->load(inFile);
    }

    inFile.read(reinterpret_cast<char*>(&n_trees), sizeof(n_trees));
    inFile.read(reinterpret_cast<char*>(&max_depth), sizeof(max_depth));
    inFile.read(reinterpret_cast<char*>(&learning_rate), sizeof(learning_rate));
    
    inFile.close();
}
