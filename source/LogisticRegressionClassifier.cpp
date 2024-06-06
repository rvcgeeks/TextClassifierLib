
#include "LogisticRegressionClassifier.h"

#include <fstream>
#include <iostream>

LogisticRegressionClassifier::LogisticRegressionClassifier()
{
    CV.setBinary(false);
    CV.setCaseSensitive(false);
    CV.setIncludeStopWords(false);
    bias = 0.0;
    epochs = 5;
    learning_rate = 0.01;
}

LogisticRegressionClassifier::~LogisticRegressionClassifier()
{
}

double LogisticRegressionClassifier::sigmoid(double z) const
{
    return 1.0 / (1.0 + exp(-z));
}

double LogisticRegressionClassifier::predict_proba(const std::vector<int>& features) const
{
    double z = bias;
    for (size_t i = 0; i < features.size(); ++i)
    {
        z += weights[i] * features[i];
    }
    return sigmoid(z);
}

void LogisticRegressionClassifier::fit(std::string abs_filepath_to_features, std::string abs_filepath_to_labels)
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
    }
    label_file.close();

    for (int epoch = 0; epoch < epochs; ++epoch)
    {
        double total_loss = 0.0;
        for (size_t i = 0; i < sentences.size(); ++i)
        {
            const auto& sentence_map = sentences[i]->sentence_map;
            std::vector<int> features(num_features, 0);
            for (const auto& entry : sentence_map)
            {
                features[entry.first] = entry.second;
            }

            double y_true = labels[i];
            double y_pred = predict_proba(features);
            double error = y_pred - y_true;
            total_loss += y_true * log(y_pred) + (1 - y_true) * log(1 - y_pred);

            for (size_t j = 0; j < features.size(); ++j)
            {
                weights[j] -= learning_rate * error * features[j];
            }
            bias -= learning_rate * error;
        }
        total_loss = -total_loss / sentences.size();
        if (epoch % 100 == 0)
        {
            std::cout << "Epoch " << epoch << " Loss: " << total_loss << std::endl;
        }
    }
}

Prediction LogisticRegressionClassifier::predict(std::string sentence)
{
    GlobalData vars;
    Prediction result;
    std::vector<string> processed_input = CV.buildSentenceVector(sentence);
    std::vector<int> feature_vector = CV.getSentenceFeatures(processed_input);
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

void LogisticRegressionClassifier::predict(std::string abs_filepath_to_features, std::string abs_filepath_to_labels)
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

void LogisticRegressionClassifier::save(const std::string& filename) const
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

void LogisticRegressionClassifier::load(const std::string& filename)
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