
#include "LogisticRegressionClassifier.h"

#include <fstream>
#include <iostream>

LogisticRegressionClassifier::LogisticRegressionClassifier() 
    : bias(0.0), learning_rate(0.01), epochs(5)
{
    CV.setBinary(false);
    CV.setCaseSensitive(false);
    CV.setIncludeStopWords(false);
}

LogisticRegressionClassifier::~LogisticRegressionClassifier() {}

float LogisticRegressionClassifier::sigmoid(float z) const 
{
    return 1.0 / (1.0 + std::exp(-z));
}

void LogisticRegressionClassifier::fit(string abs_filepath_to_features, string abs_filepath_to_labels) 
{
    CV.fit(abs_filepath_to_features, abs_filepath_to_labels);
    
    unsigned int vocab_size = CV.getWordArraySize();
    weights.resize(vocab_size, 0.0);

    std::cout << "Fitting LogisticRegressionClassifier..." << std::endl;
    
    for (int epoch = 0; epoch < epochs; ++epoch) 
    {
        for (const auto& sentence : CV.sentences) 
        {
            update_weights(sentence->sentence_array, sentence->label);
        }
    }
}

void LogisticRegressionClassifier::update_weights(std::vector<int>& features, bool label) 
{
    float linear_model = bias;
    for (size_t i = 0; i < features.size(); ++i) 
    {
        linear_model += weights[i] * features[i];
    }
    float prediction = sigmoid(linear_model);
    float error = label - prediction;

    bias += learning_rate * error;
    for (size_t i = 0; i < features.size(); ++i) 
    {
        weights[i] += learning_rate * error * features[i];
    }
}

int LogisticRegressionClassifier::predict(string sentence) 
{
    GlobalData vars;
    std::vector<string> processed_input = CV.buildSentenceVector(sentence);
    std::vector<int> sentence_vector(CV.getWordArraySize(), 0);
    
    for (const auto& word : processed_input) 
    {
        for (size_t i = 0; i < CV.getWordArraySize(); ++i) 
        {
            if (CV.getWord(i) == word) 
            {
                sentence_vector[i]++;
            }
        }
    }

    float linear_model = bias;
    for (size_t i = 0; i < sentence_vector.size(); ++i) 
    {
        linear_model += weights[i] * sentence_vector[i];
    }
    float prediction = sigmoid(linear_model);
    
    if (prediction > 0.5) 
    {
        return vars.POS;
    } 
    else if(prediction < 0.5) 
    {
        return vars.NEG;
    }
	else
    {
        return vars.NEU;
    }
}

void LogisticRegressionClassifier::predict(string abs_filepath_to_features, string abs_filepath_to_labels) 
{
    std::ifstream in(abs_filepath_to_features);
    std::ofstream out(abs_filepath_to_labels);
    if (!in || !out) 
    {
        std::cerr << "ERROR: Cannot open input or output file.\n";
        return;
    }

    std::string feature_input;
    while (getline(in, feature_input)) 
    {
        int label_output = predict(feature_input);
        out << label_output << std::endl;
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

    size_t weights_size = weights.size();
    outFile.write(reinterpret_cast<const char*>(&weights_size), sizeof(weights_size));
    outFile.write(reinterpret_cast<const char*>(weights.data()), weights.size() * sizeof(float));
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

    size_t weights_size;
    inFile.read(reinterpret_cast<char*>(&weights_size), sizeof(weights_size));
    weights.resize(weights_size);
    inFile.read(reinterpret_cast<char*>(weights.data()), weights_size * sizeof(float));
    inFile.read(reinterpret_cast<char*>(&bias), sizeof(bias));

    inFile.close();
}
