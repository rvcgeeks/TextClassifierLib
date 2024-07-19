
/*++

Revision History:
	Date:	Jun 28, 2024.
	Author:	Rajas Chavadekar.
	Desc:	Created.

--*/

#include "LogisticRegressionClassifier.h"

#include <fstream>
#include <iostream>
#include <sstream>
#include <cmath>
#include <unordered_map>  // TR1 unordered_map

LogisticRegressionClassifier::LogisticRegressionClassifier(BaseVectorizer* pvec)
{
    pVec = pvec;
}

LogisticRegressionClassifier::~LogisticRegressionClassifier()
{
    delete pVec;
}

double LogisticRegressionClassifier::predict_proba(const std::vector<double>& features) const
{
    double z = bias;
    for (size_t i = 0; i < features.size(); ++i) {
        z += weights[i] * features[i];
    }

    return 1.0 / (1.0 + exp(-z));
}

void LogisticRegressionClassifier::setHyperparameters(std::string hyperparameters)
{
    std::string token;
    std::istringstream tokenStream(hyperparameters);

    // "bias=0.0,epochs=15,learning_rate=0.01,l1_regularization_param=0.005,l2_regularization_param=0.0"
    bias = 0.0;
    epochs = 15;
    learning_rate = 0.01;
    l1_regularization_param = 0.005;
    l2_regularization_param = 0.0;

    while (std::getline(tokenStream, token, ',')) {
        std::istringstream pairStream(token);
        std::string key;
        double value;

        if (std::getline(pairStream, key, '=') && pairStream >> value) {
            std::cout << key << " = " << value << std::endl;
            if (key == "minfrequency") {
                minfrequency = value;
            }
            if (key == "bias") {
                bias = value;
            }
            else if (key == "epochs") {
                epochs = static_cast<int>(value);  // Ensure integer conversion
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

void LogisticRegressionClassifier::fit(std::string abs_filepath_to_features, std::string abs_filepath_to_labels)
{
    if (minfrequency > 0)
    {
        pVec->scanForSparseHistogram(abs_filepath_to_features, minfrequency);
    }
    pVec->fit(abs_filepath_to_features, abs_filepath_to_labels);

    size_t num_features = pVec->word_array.size();
    weights.assign(num_features, 0.0);

	std::vector<std::tr1::shared_ptr<Sentence> > sentences = pVec->sentences;
    std::vector<int> labels(sentences.size());

    std::ifstream label_file(abs_filepath_to_labels.c_str());
    if (!label_file.is_open()) {
        throw std::runtime_error("Unable to open label file");
    }
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
            std::vector<double> features;
            const std::tr1::unordered_map<int, double>& sentence_map = sentences[i]->sentence_map;
            features = pVec->getFrequencies(sentence_map);
            double y_true = labels[i];
            double y_false;
            if (1.0 == y_true)
            {
                y_false = 0.0;
            }
            else
            {
                y_false = 1.0;
            }
            double y_pred = predict_proba(features);
            double error = y_pred - y_true;

            total_loss += y_true * log(y_pred) + y_false * log(1 - y_pred);

            for (size_t j = 0; j < features.size(); ++j)
            {
                double gradient = error * features[j];

                weights[j] -= learning_rate * (gradient + l1_regularization_param * (weights[j] > 0 ? 1 : -1) + 2 * l2_regularization_param * weights[j]);
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

Prediction LogisticRegressionClassifier::predict(std::string sentence, bool preprocess)
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

void LogisticRegressionClassifier::predict(std::string abs_filepath_to_features, std::string abs_filepath_to_labels, bool preprocess)
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

void LogisticRegressionClassifier::save(const std::string& filename) const
{
    std::ofstream outFile(filename.c_str(), std::ios::binary);
    if (!outFile.is_open())
    {
        std::cerr << "Failed to open file for writing." << std::endl;
        return;
    }

    pVec->save(outFile);

    size_t weight_size = weights.size();
    outFile.write(reinterpret_cast<const char*>(&weight_size), sizeof(weight_size));
    outFile.write(reinterpret_cast<const char*>(&weights[0]), weight_size * sizeof(double));
    outFile.write(reinterpret_cast<const char*>(&bias), sizeof(bias));

    outFile.close();
}

void LogisticRegressionClassifier::load(const std::string& filename)
{
    std::ifstream inFile(filename.c_str(), std::ios::binary);
    if (!inFile.is_open())
    {
        std::cerr << "Failed to open file for reading." << std::endl;
        return;
    }

    pVec->load(inFile);

    size_t weight_size;
    inFile.read(reinterpret_cast<char*>(&weight_size), sizeof(weight_size));
    weights.resize(weight_size);
    inFile.read(reinterpret_cast<char*>(&weights[0]), weight_size * sizeof(double));
    inFile.read(reinterpret_cast<char*>(&bias), sizeof(bias));

    inFile.close();
}
