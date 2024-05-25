
#include "SVCClassifier.h"

#include <fstream>
#include <iostream>
#include "GlobalData.h"

SVCClassifier::SVCClassifier()
{
    CV.setBinary(false);
    CV.setCaseSensitive(false);
    CV.setIncludeStopWords(false);
    bias = 0.0;
    epochs = 50;
    learning_rate = 0.05;
    regularization_param = 0.01;
}

SVCClassifier::~SVCClassifier()
{
}

double SVCClassifier::predict_margin(const std::vector<int>& features) const
{
    double margin = bias;
    for (size_t i = 0; i < features.size(); ++i)
    {
        margin += weights[i] * features[i];
    }
    return margin;
}

void SVCClassifier::fit(std::string abs_filepath_to_features, std::string abs_filepath_to_labels)
{
    CV.fit(abs_filepath_to_features, abs_filepath_to_labels);

    size_t num_features = CV.word_array.size();
    weights.assign(num_features, 0.0);

    std::vector<std::shared_ptr<Sentence>> sentences = CV.sentences;
    std::vector<int> labels(sentences.size());

    std::ifstream label_file(abs_filepath_to_labels);
    std::string label;
    for (size_t i = 0; i < labels.size(); ++i)
    {
        label_file >> labels[i];
        // Convert labels to +1 or -1 for SVM
        labels[i] = labels[i] == 1 ? 1 : -1;
    }
    label_file.close();

    for (int epoch = 0; epoch < epochs; ++epoch)
    {
        for (size_t i = 0; i < sentences.size(); ++i)
        {
            const auto& sentence_map = sentences[i]->sentence_map;
            std::vector<int> features(num_features, 0);
            for (const auto& entry : sentence_map)
            {
                features[entry.first] = entry.second;
            }

            double y_true = labels[i];
            double margin = predict_margin(features);

            if (y_true * margin < 1)
            {
                for (size_t j = 0; j < features.size(); ++j)
                {
                    weights[j] += learning_rate * (y_true * features[j] - 2 * regularization_param * weights[j]);
                }
                bias += learning_rate * y_true;
            }
            else
            {
                for (size_t j = 0; j < features.size(); ++j)
                {
                    weights[j] += learning_rate * (-2 * regularization_param * weights[j]);
                }
            }
        }
    }
}

int SVCClassifier::predict(std::string sentence)
{
    GlobalData vars;
    std::vector<string> processed_input = CV.buildSentenceVector(sentence);
    std::vector<int> feature_vector = CV.getSentenceFeatures(processed_input);
    double margin = predict_margin(feature_vector);

    if (margin > 0)
    {
        return vars.POS;
    }
    else if (margin < 0)
    {
        return vars.NEG;
    }
    else
    {
        return vars.NEU;
    }
}

void SVCClassifier::predict(std::string abs_filepath_to_features, std::string abs_filepath_to_labels)
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

void SVCClassifier::save(const std::string& filename) const
{
    std::ofstream outFile(filename, std::ios::binary);
    if (!outFile.is_open())
    {
        std::cerr << "Failed to open file for writing." << std::endl;
        return;
    }

    CV.save(outFile);

    size_t weight_size = weights.size();
    outFile.write(reinterpret_cast<const char*>(&weight_size), sizeof(weight_size));
    outFile.write(reinterpret_cast<const char*>(weights.data()), weight_size * sizeof(double));
    outFile.write(reinterpret_cast<const char*>(&bias), sizeof(bias));

    outFile.close();
}

void SVCClassifier::load(const std::string& filename)
{
    std::ifstream inFile(filename, std::ios::binary);
    if (!inFile.is_open())
    {
        std::cerr << "Failed to open file for reading." << std::endl;
        return;
    }

    CV.load(inFile);

    size_t weight_size;
    inFile.read(reinterpret_cast<char*>(&weight_size), sizeof(weight_size));
    weights.resize(weight_size);
    inFile.read(reinterpret_cast<char*>(weights.data()), weight_size * sizeof(double));
    inFile.read(reinterpret_cast<char*>(&bias), sizeof(bias));

    inFile.close();
}
