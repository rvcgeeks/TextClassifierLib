
#include "RandomForestClassifier.h"

#include <fstream>
#include <iostream>
#include <algorithm>

RandomForestClassifier::RandomForestClassifier(int num_trees, int max_depth)
    : num_trees(num_trees), max_depth(max_depth)
{
    CV.setBinary(false);
    CV.setCaseSensitive(false);
    CV.setIncludeStopWords(false);
}

RandomForestClassifier::~RandomForestClassifier()
{
}

void RandomForestClassifier::fit(std::string abs_filepath_to_features, std::string abs_filepath_to_labels)
{
    CV.fit(abs_filepath_to_features, abs_filepath_to_labels);
    std::vector<std::shared_ptr<Sentence>> sentences = CV.sentences;
    
    for (int i = 0; i < num_trees; ++i)
    {
        auto tree = std::make_shared<DecisionTree>(max_depth);
        tree->fit(CV, sentences);
        trees.push_back(tree);
    }
}

int RandomForestClassifier::predict(std::string sentence)
{
    GlobalData vars;
    std::vector<std::string> processed_input = CV.buildSentenceVector(sentence);
    std::vector<int> feature_vector = CV.getSentenceFeatures(processed_input);
    
    std::vector<int> votes(3, 0); // Assuming 3 classes: POS, NEG, NEU
    for (const auto& tree : trees)
    {
        int prediction = tree->predict(feature_vector);
        votes[prediction]++;
    }
    
    return std::distance(votes.begin(), std::max_element(votes.begin(), votes.end()));
}

void RandomForestClassifier::predict(std::string abs_filepath_to_features, std::string abs_filepath_to_labels)
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

void RandomForestClassifier::save(const std::string& filename) const
{
    std::ofstream outFile(filename, std::ios::binary);
    if (!outFile.is_open())
    {
        std::cerr << "Failed to open file for writing." << std::endl;
        return;
    }

    CV.save(outFile);

    size_t num_trees = trees.size();
    outFile.write(reinterpret_cast<const char*>(&num_trees), sizeof(num_trees));
    for (const auto& tree : trees)
    {
        tree->save(outFile);
    }

    outFile.close();
}

void RandomForestClassifier::load(const std::string& filename)
{
    std::ifstream inFile(filename, std::ios::binary);
    if (!inFile.is_open())
    {
        std::cerr << "Failed to open file for reading." << std::endl;
        return;
    }

    CV.load(inFile);

    size_t num_trees;
    inFile.read(reinterpret_cast<char*>(&num_trees), sizeof(num_trees));
    trees.resize(num_trees);
    for (size_t i = 0; i < num_trees; ++i)
    {
        auto tree = std::make_shared<DecisionTree>();
        tree->load(inFile);
        trees[i] = tree;
    }

    inFile.close();
}
