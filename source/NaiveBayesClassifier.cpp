
#include "NaiveBayesClassifier.h"

#include <fstream>
#include <iostream>
#include <cmath>

NaiveBayesClassifier::NaiveBayesClassifier()
{
    CV.setBinary(false);
    CV.setCaseSensitive(false);
    CV.setIncludeStopWords(false);
}

NaiveBayesClassifier::~NaiveBayesClassifier()
{
}

void NaiveBayesClassifier::fit(std::string abs_filepath_to_features, std::string abs_filepath_to_labels)
{
    CV.fit(abs_filepath_to_features, abs_filepath_to_labels);

    std::vector<std::shared_ptr<Sentence>> sentences = CV.sentences;
    int num_sentences = sentences.size();

    int num_pos = 0;
    int num_neg = 0;
    std::unordered_map<int, int> word_count_pos;
    std::unordered_map<int, int> word_count_neg;
    int total_words_pos = 0;
    int total_words_neg = 0;

    for (const auto& sentence : sentences)
    {
        if (sentence->label)
        {
            num_pos++;
            for (const auto& entry : sentence->sentence_map)
            {
                word_count_pos[entry.first] += entry.second;
                total_words_pos += entry.second;
            }
        }
        else
        {
            num_neg++;
            for (const auto& entry : sentence->sentence_map)
            {
                word_count_neg[entry.first] += entry.second;
                total_words_neg += entry.second;
            }
        }
    }

    log_prior_pos = std::log(static_cast<double>(num_pos) / num_sentences);
    log_prior_neg = std::log(static_cast<double>(num_neg) / num_sentences);

    for (const auto& word : CV.word_array)
    {
        int idx = CV.word_to_idx[word];
        log_prob_pos[idx] = std::log((word_count_pos[idx] + 1.0) / (total_words_pos + CV.getWordArraySize()));
        log_prob_neg[idx] = std::log((word_count_neg[idx] + 1.0) / (total_words_neg + CV.getWordArraySize()));
    }
}

double NaiveBayesClassifier::calculate_log_probability(const std::vector<int>& features, bool is_positive) const
{
    double log_prob = is_positive ? log_prior_pos : log_prior_neg;
    const auto& log_prob_map = is_positive ? log_prob_pos : log_prob_neg;

    for (size_t i = 0; i < features.size(); ++i)
    {
        if (features[i] > 0)
        {
            log_prob += features[i] * log_prob_map.at(i);
        }
    }
    return log_prob;
}

int NaiveBayesClassifier::predict(std::string sentence)
{
    GlobalData vars;
    std::vector<std::string> processed_input = CV.buildSentenceVector(sentence);
    std::vector<int> feature_vector = CV.getSentenceFeatures(processed_input);

    double log_prob_pos = calculate_log_probability(feature_vector, true);
    double log_prob_neg = calculate_log_probability(feature_vector, false);

    if (log_prob_pos > log_prob_neg)
    {
        return vars.POS;
    }
    else if (log_prob_neg > log_prob_pos)
    {
        return vars.NEG;
    }
    else
    {
        return vars.NEU;
    }
}

void NaiveBayesClassifier::predict(std::string abs_filepath_to_features, std::string abs_filepath_to_labels)
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

void NaiveBayesClassifier::save(const std::string& filename) const
{
    std::ofstream outFile(filename, std::ios::binary);
    if (!outFile.is_open())
    {
        std::cerr << "Failed to open file for writing." << std::endl;
        return;
    }

    CV.save(outFile);

    size_t size;

    size = log_prob_pos.size();
    outFile.write(reinterpret_cast<const char*>(&size), sizeof(size));
    for (const auto& pair : log_prob_pos)
    {
        outFile.write(reinterpret_cast<const char*>(&pair.first), sizeof(pair.first));
        outFile.write(reinterpret_cast<const char*>(&pair.second), sizeof(pair.second));
    }

    size = log_prob_neg.size();
    outFile.write(reinterpret_cast<const char*>(&size), sizeof(size));
    for (const auto& pair : log_prob_neg)
    {
        outFile.write(reinterpret_cast<const char*>(&pair.first), sizeof(pair.first));
        outFile.write(reinterpret_cast<const char*>(&pair.second), sizeof(pair.second));
    }

    outFile.write(reinterpret_cast<const char*>(&log_prior_pos), sizeof(log_prior_pos));
    outFile.write(reinterpret_cast<const char*>(&log_prior_neg), sizeof(log_prior_neg));

    outFile.close();
}

void NaiveBayesClassifier::load(const std::string& filename)
{
    std::ifstream inFile(filename, std::ios::binary);
    if (!inFile.is_open())
    {
        std::cerr << "Failed to open file for reading." << std::endl;
        return;
    }

    CV.load(inFile);

    size_t size;

    inFile.read(reinterpret_cast<char*>(&size), sizeof(size));
    log_prob_pos.clear();
    for (size_t i = 0; i < size; ++i)
    {
        int key;
        double value;
        inFile.read(reinterpret_cast<char*>(&key), sizeof(key));
        inFile.read(reinterpret_cast<char*>(&value), sizeof(value));
        log_prob_pos[key] = value;
    }

    inFile.read(reinterpret_cast<char*>(&size), sizeof(size));
    log_prob_neg.clear();
    for (size_t i = 0; i < size; ++i)
    {
        int key;
        double value;
        inFile.read(reinterpret_cast<char*>(&key), sizeof(key));
        inFile.read(reinterpret_cast<char*>(&value), sizeof(value));
        log_prob_neg[key] = value;
    }

    inFile.read(reinterpret_cast<char*>(&log_prior_pos), sizeof(log_prior_pos));
    inFile.read(reinterpret_cast<char*>(&log_prior_neg), sizeof(log_prior_neg));

    inFile.close();
}
