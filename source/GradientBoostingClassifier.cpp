
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
        score += learning_rate * (tree_prediction - l1_regularization_param * std::abs(tree_prediction) - l2_regularization_param * (tree_prediction * tree_prediction));
    }
    return 1.0 / (1.0 + exp(-score));
}

void GradientBoostingClassifier::setHyperparameters(std::string hyperparameters)
{
    std::string token;
    std::istringstream tokenStream(hyperparameters);

    // "n_trees=50,max_depth=5,learning_rate=0.01,l1_regularization_param=0.005,l2_regularization_param=0.0"
    n_trees = 50;
    max_depth = 5;
    learning_rate = 0.01;
    l1_regularization_param = 0.005;
    l2_regularization_param = 0.0;

    while (std::getline(tokenStream, token, ',')) {
        std::istringstream pairStream(token);
        std::string key;
        double value;

        if (std::getline(pairStream, key, '=') && pairStream >> value) {
            cout << key << " = " << value << endl;
            if (key == "n_trees") {
                n_trees = value;
            }
            else if (key == "max_depth") {
                max_depth = value;
            }
            else if (key == "learning_rate") {
                learning_rate = value;
            }
            else if (key == "l1_regularization_param") {
                l1_regularization_param = value;
            }
            else if (key == "l2_regularization_param") {
                l2_regularization_param = value;
            }
        }
    }
}

void GradientBoostingClassifier::fit(std::string abs_filepath_to_features, std::string abs_filepath_to_labels)
{
    pVec->fit(abs_filepath_to_features, abs_filepath_to_labels);

    size_t num_features = pVec->word_array.size();
    trees.clear();
    tree_weights.clear();

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
            const auto& sentence_map = sentences[j]->sentence_map;
            std::vector<double> features(num_features, 0);
            for (const auto& entry : sentence_map)
            {
                features[entry.first] = entry.second;
            }

            double y_true = labels[j];
            double y_pred = predict_proba(features);
            residuals[j] = y_true - y_pred;
        }

        auto tree = std::make_unique<DecisionTree>(max_depth);
        tree->fit(sentences);
        trees.push_back(std::move(tree));
        tree_weights.push_back(learning_rate);
    }
}

Prediction GradientBoostingClassifier::predict(std::string sentence)
{
    GlobalData vars;
    Prediction result;
    std::vector<std::string> processed_input = pVec->buildSentenceVector(sentence);
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
        Prediction result = predict(feature_input);
        out << result.label << "," << result.probability << std::endl;
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
