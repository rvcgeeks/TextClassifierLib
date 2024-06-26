
#include "KNNClassifier.h"

#include <fstream>
#include <iostream>
#include <algorithm>
#include <cmath>

KNNClassifier::KNNClassifier(BaseVectorizer* pvec)
{
    pVec = pvec;
}

KNNClassifier::~KNNClassifier()
{
    delete pVec;
}

void KNNClassifier::setHyperparameters(std::string hyperparameters)
{
    std::string token;
    std::istringstream tokenStream(hyperparameters);

    // "k=3"
    k = 3;

    while (std::getline(tokenStream, token, ',')) {
        std::istringstream pairStream(token);
        std::string key;
        double value;

        if (std::getline(pairStream, key, '=') && pairStream >> value) {
            cout << key << " = " << value << endl;
            if (key == "minfrequency") {
                minfrequency = value;
            }
            if (key == "k") {
                k = value;
            }
        }
    }
}

void KNNClassifier::fit(std::string abs_filepath_to_features, std::string abs_filepath_to_labels)
{
    if (minfrequency > 0)
    {
        pVec->scanForSparseHistogram(abs_filepath_to_features, minfrequency);
    }
    pVec->fit(abs_filepath_to_features, abs_filepath_to_labels);

    size_t num_features = pVec->word_array.size();
    std::vector<std::shared_ptr<Sentence>> sentences = pVec->sentences;
    training_features.clear();
    training_labels.clear();

    std::ifstream label_file(abs_filepath_to_labels);
    std::string label;
    for (size_t i = 0; i < sentences.size(); ++i)
    {
        std::vector<double> features;
        const auto& sentence_map = sentences[i]->sentence_map;
        features = pVec->getFrequencies(sentence_map);
        training_features.push_back(features);

        label_file >> label;
        training_labels.push_back(std::stoi(label));
    }
    label_file.close();

    kd_tree.build(training_features, training_labels);
}

void KNNClassifier::predict(std::string abs_filepath_to_features, std::string abs_filepath_to_labels, bool preprocess)
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

Prediction KNNClassifier::predict(std::string sentence, bool preprocess)
{
    GlobalData vars;
    std::vector<std::string> processed_input = pVec->buildSentenceVector(sentence, preprocess);
    std::vector<double> feature_vector = pVec->getSentenceFeatures(processed_input);

    int label = kd_tree.nearestNeighbor(feature_vector);

    // Calculate probability
    double total_distance = 0.0;
    std::vector<double> closest_distances = kd_tree.getClosestDistances(feature_vector, k);
    for (double dist : closest_distances)
    {
        total_distance += dist;
    }
    double probability = 1.0 - (total_distance / closest_distances.size());

    return { label, probability };
}

int KNNClassifier::getLabel(const std::vector<double>& features) const
{
    // Deprecated: getLabel is no longer needed with KDTree nearestNeighbor.
    return 0;
}

double KNNClassifier::calculateDistance(const std::vector<double>& a, const std::vector<double>& b) const
{
    double sum = 0.0;
    for (size_t i = 0; i < a.size(); ++i)
    {
        double diff = a[i] - b[i];
        sum += diff * diff;
    }
    return std::sqrt(sum);
}

void KNNClassifier::save(const std::string& filename) const
{
    std::ofstream outFile(filename, std::ios::binary);
    if (!outFile.is_open())
    {
        std::cerr << "Failed to open file for writing." << std::endl;
        return;
    }

    pVec->save(outFile);

    size_t training_features_size = training_features.size();
    outFile.write(reinterpret_cast<const char*>(&training_features_size), sizeof(training_features_size));
    for (const auto& features : training_features)
    {
        size_t features_size = features.size();
        outFile.write(reinterpret_cast<const char*>(&features_size), sizeof(features_size));
        outFile.write(reinterpret_cast<const char*>(features.data()), features_size * sizeof(int));
    }

    size_t training_labels_size = training_labels.size();
    outFile.write(reinterpret_cast<const char*>(&training_labels_size), sizeof(training_labels_size));
    outFile.write(reinterpret_cast<const char*>(training_labels.data()), training_labels_size * sizeof(int));

    outFile.close();
}

void KNNClassifier::load(const std::string& filename)
{
    std::ifstream inFile(filename, std::ios::binary);
    if (!inFile.is_open())
    {
        std::cerr << "Failed to open file for reading." << std::endl;
        return;
    }

    pVec->load(inFile);

    size_t training_features_size;
    inFile.read(reinterpret_cast<char*>(&training_features_size), sizeof(training_features_size));
    training_features.resize(training_features_size);
    for (auto& features : training_features)
    {
        size_t features_size;
        inFile.read(reinterpret_cast<char*>(&features_size), sizeof(features_size));
        features.resize(features_size);
        inFile.read(reinterpret_cast<char*>(features.data()), features_size * sizeof(int));
    }

    size_t training_labels_size;
    inFile.read(reinterpret_cast<char*>(&training_labels_size), sizeof(training_labels_size));
    training_labels.resize(training_labels_size);
    inFile.read(reinterpret_cast<char*>(training_labels.data()), training_labels_size * sizeof(int));

    inFile.close();

    kd_tree.build(training_features, training_labels);
}
