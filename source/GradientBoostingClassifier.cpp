
#include "GradientBoostingClassifier.h"

#include <fstream>
#include <iostream>

GradientBoostingClassifier::GradientBoostingClassifier(BaseVectorizer* pvec)
{
	pVec = pvec;
}

GradientBoostingClassifier::~GradientBoostingClassifier()
{
    delete pVec;
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
            cout << key << " = " << value << endl;
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

    std::vector<std::shared_ptr<Sentence>> sentences = pVec->sentences;
    std::vector<double> labels(sentences.size());

    std::ifstream label_file(abs_filepath_to_labels);
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
            const auto& sentence_map = sentences[j]->sentence_map;
            features = pVec->getFrequencies(sentence_map);
            double y_true = labels[j];
            double y_pred = predict_proba(features);
            residuals[j] = y_true - y_pred;
        }

        auto tree = std::make_unique<DecisionTree>(max_depth);
        tree->fit(sentences);
        trees.push_back(std::move(tree));
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
    std::ifstream in(abs_filepath_to_features);
    std::ofstream out(abs_filepath_to_labels);
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
        auto start = std::chrono::high_resolution_clock::now();
        #endif

        Prediction result = predict(feature_input, preprocess);

        #ifdef BENCHMARK
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> duration = end - start;
        double milliseconds = duration.count();
        sumduration += milliseconds;
        sumstrlen += feature_input.length();
        num_rows++;
        #endif

        out << result.label << "," << result.probability << std::endl;
    }

    #ifdef BENCHMARK
    double avgduration = sumduration / num_rows;
    cout << "Average Time per Text = " << avgduration << " ms" << endl;
    double avgstrlen = sumstrlen / num_rows;
    cout << "Average Length of Text (chars) = " << avgstrlen << endl;
    #endif

    in.close();
    out.close();
}

void GradientBoostingClassifier::save(const std::string& filename) const
{
    std::ofstream outFile(filename, std::ios::binary);
    if (!outFile.is_open())
    {
        std::cerr << "Failed to open file for writing." << std::endl;
        return;
    }

    pVec->save(outFile);

    size_t tree_count = trees.size();
    outFile.write(reinterpret_cast<const char*>(&tree_count), sizeof(tree_count));

    for (const auto& tree : trees)
    {
        tree->save(outFile);
    }

    outFile.write(reinterpret_cast<const char*>(&n_trees), sizeof(n_trees));
    outFile.write(reinterpret_cast<const char*>(&max_depth), sizeof(max_depth));
    outFile.write(reinterpret_cast<const char*>(&learning_rate), sizeof(learning_rate));
    
    outFile.close();
}

void GradientBoostingClassifier::load(const std::string& filename)
{
    std::ifstream inFile(filename, std::ios::binary);
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
        auto tree = std::make_unique<DecisionTree>(max_depth);
        tree->load(inFile);
        trees[i] = std::move(tree);
    }

    inFile.read(reinterpret_cast<char*>(&n_trees), sizeof(n_trees));
    inFile.read(reinterpret_cast<char*>(&max_depth), sizeof(max_depth));
    inFile.read(reinterpret_cast<char*>(&learning_rate), sizeof(learning_rate));
    
    inFile.close();
}
