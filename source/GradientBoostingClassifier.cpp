
#include "GradientBoostingClassifier.h"

#include <fstream>
#include <iostream>

GradientBoostingClassifier::GradientBoostingClassifier()
    : n_trees(100), max_depth(3), learning_rate(0.1)
{
    CV.setBinary(false);
    CV.setCaseSensitive(false);
    CV.setIncludeStopWords(false);
}

GradientBoostingClassifier::~GradientBoostingClassifier()
{
}

double GradientBoostingClassifier::predict_tree(const DecisionTree& tree, const std::vector<int>& features) const
{
    return tree.predict(features);
}

double GradientBoostingClassifier::predict_proba(const std::vector<int>& features) const
{
    double score = 0.0;
    for (size_t i = 0; i < trees.size(); ++i)
    {
        score += learning_rate * predict_tree(*trees[i], features);
    }
    return 1.0 / (1.0 + exp(-score));
}

void GradientBoostingClassifier::fit(std::string abs_filepath_to_features, std::string abs_filepath_to_labels)
{
    CV.fit(abs_filepath_to_features, abs_filepath_to_labels);

    size_t num_features = CV.word_array.size();
    trees.clear();
    tree_weights.clear();

    std::vector<std::shared_ptr<Sentence>> sentences = CV.sentences;
    std::vector<int> labels(sentences.size());

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
            const auto& sentence_map = sentences[j]->sentence_map;
            std::vector<int> features(num_features, 0);
            for (const auto& entry : sentence_map)
            {
                features[entry.first] = entry.second;
            }

            double y_true = labels[j];
            double y_pred = predict_proba(features);
            residuals[j] = y_true - y_pred;
        }

        auto tree = std::make_unique<DecisionTree>(max_depth);
        tree->fit(CV, sentences);
        trees.push_back(std::move(tree));
        tree_weights.push_back(learning_rate);
    }
}

int GradientBoostingClassifier::predict(std::string sentence)
{
    GlobalData vars;
    std::vector<std::string> processed_input = CV.buildSentenceVector(sentence);
    std::vector<int> feature_vector = CV.getSentenceFeatures(processed_input);
    double probability = predict_proba(feature_vector);

    if (probability > 0.5)
    {
        return vars.POS;
    }
    else
    {
        return vars.NEG;
    }
}

void GradientBoostingClassifier::predict(std::string abs_filepath_to_features, std::string abs_filepath_to_labels)
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

    while (getline(in, feature_input))
    {
        int label_output = predict(feature_input);
        out << label_output << std::endl;
    }

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

    CV.save(outFile);

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

    CV.load(inFile);

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
