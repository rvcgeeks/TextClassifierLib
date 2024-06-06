
#include "RandomForestClassifier.h"

#include <fstream>
#include <iostream>
#include <algorithm>

RandomForestClassifier::RandomForestClassifier(int vectorizerid, int num_trees, int max_depth)
    : num_trees(num_trees), max_depth(max_depth)
{
    switch (vectorizerid)
    {
        case ID_VECTORIZER_COUNT:
            pVec = new CountVectorizer();
            break;

        case ID_VECTORIZER_TFIDF:
            pVec = new TfidfVectorizer();
            break;

        default:
            throw runtime_error("Unknown Vectorizer!");
    }

    pVec->setBinary(false);
    pVec->setCaseSensitive(false);
    pVec->setIncludeStopWords(false);
}

RandomForestClassifier::~RandomForestClassifier()
{
    delete pVec;
}

void RandomForestClassifier::fit(std::string abs_filepath_to_features, std::string abs_filepath_to_labels)
{
    pVec->fit(abs_filepath_to_features, abs_filepath_to_labels);
    std::vector<std::shared_ptr<Sentence>> sentences = pVec->sentences;
    
    for (int i = 0; i < num_trees; ++i)
    {
        auto tree = std::make_shared<DecisionTree>(max_depth);
        tree->fit(sentences);
        trees.push_back(tree);
    }
}

Prediction RandomForestClassifier::predict(std::string sentence)
{
    GlobalData vars;
    Prediction result;

    std::vector<std::string> processed_input = pVec->buildSentenceVector(sentence);
    std::vector<double> feature_vector = pVec->getSentenceFeatures(processed_input);

    std::vector<int> votes(3, 0); // Assuming 3 classes: POS, NEG, NEU
    for (const auto& tree : trees)
    {
        int prediction = tree->predict(feature_vector).label;
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
        Prediction result = predict(feature_input);
        out << result.label << "," << result.probability << std::endl;
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

    pVec->save(outFile);

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

    pVec->load(inFile);

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
