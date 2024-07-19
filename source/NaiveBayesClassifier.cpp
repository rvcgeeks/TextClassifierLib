
/*++

Revision History:
	Date:	Jun 28, 2024.
	Author:	Rajas Chavadekar.
	Desc:	Created.

--*/

#include "NaiveBayesClassifier.h"

#include <fstream>
#include <iostream>
#include <cmath>

NaiveBayesClassifier::NaiveBayesClassifier(BaseVectorizer* pvec)
{
    pVec = pvec;
}

NaiveBayesClassifier::~NaiveBayesClassifier()
{
    delete pVec;
}

void NaiveBayesClassifier::setHyperparameters(std::string hyperparameters)
{
    std::string token;
    std::istringstream tokenStream(hyperparameters);

    // "smoothing_param_m=1.0,smoothing_param_p=0.5"
    smoothing_param_m = 1.0;
	smoothing_param_p = 0.5;

    while (std::getline(tokenStream, token, ',')) {
        std::istringstream pairStream(token);
        std::string key;
        double value;

        if (std::getline(pairStream, key, '=') && pairStream >> value) {
            cout << key << " = " << value << endl;        
            if (key == "minfrequency") {
                minfrequency = value;
            }
            if (key == "smoothing_param_m") {
                smoothing_param_m = value;
            }
			else if (key == "smoothing_param_p") {
                smoothing_param_p = value;
            }
        }
    }
}

void NaiveBayesClassifier::fit(std::string abs_filepath_to_features, std::string abs_filepath_to_labels)
{
    if (minfrequency > 0)
    {
        pVec->scanForSparseHistogram(abs_filepath_to_features, minfrequency);
    }
    pVec->fit(abs_filepath_to_features, abs_filepath_to_labels);

    std::vector<std::shared_ptr<Sentence>> sentences = pVec->sentences;
    int num_sentences = sentences.size();

    int num_pos = 0;
    int num_neg = 0;
    std::unordered_map<int, double> word_count_pos;
    std::unordered_map<int, double> word_count_neg;
    std::vector<double> tfidf_features_pos;
    std::vector<double> tfidf_features_neg;
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
	double mp = smoothing_param_m * smoothing_param_p;
    double tfidf_sum_pos = 0.0;
    double tfidf_sum_neg = 0.0;

    if (ID_VECTORIZER_TFIDF == pVec->this_vectorizer_id) {
        tfidf_features_pos = pVec->getFrequencies(word_count_pos);
        tfidf_features_neg = pVec->getFrequencies(word_count_neg);
        for (const auto& v : tfidf_features_pos)
        {
            tfidf_sum_pos += v;
        }
        for (const auto& v : tfidf_features_neg)
        {
            tfidf_sum_neg += v;
        }
    }

    for (const auto& word : pVec->word_array)
    {
        int idx = pVec->word_to_idx[word];
        if (ID_VECTORIZER_TFIDF == pVec->this_vectorizer_id) {
            log_prob_pos[idx] = std::log((tfidf_features_pos[idx] + mp) / (tfidf_sum_pos + smoothing_param_m + pVec->word_array.size()));
            log_prob_neg[idx] = std::log((tfidf_features_neg[idx] + mp) / (tfidf_sum_neg + smoothing_param_m + pVec->word_array.size()));
        }
        else 
        {
            log_prob_pos[idx] = std::log((word_count_pos[idx] + mp) / (total_words_pos + smoothing_param_m + pVec->word_array.size()));
            log_prob_neg[idx] = std::log((word_count_neg[idx] + mp) / (total_words_neg + smoothing_param_m + pVec->word_array.size()));
        }
    }
}

double NaiveBayesClassifier::calculate_log_probability(const std::vector<double>& features, bool is_positive) const
{
    double log_prob = is_positive ? log_prior_pos : log_prior_neg;
    const auto& log_prob_map = is_positive ? log_prob_pos : log_prob_neg;

    for (size_t i = 0; i < features.size(); ++i)
    {
        if (features[i] > 0)
        {
            log_prob += std::abs(features[i]) * log_prob_map.at(i);
        }
    }
    return log_prob;
}

Prediction NaiveBayesClassifier::predict(std::string sentence, bool preprocess)
{
    GlobalData vars;
    Prediction result;
    std::vector<std::string> processed_input = pVec->buildSentenceVector(sentence, preprocess);
    std::vector<double> feature_vector = pVec->getSentenceFeatures(processed_input);

    double log_prob_pos = calculate_log_probability(feature_vector, true);
    double log_prob_neg = calculate_log_probability(feature_vector, false);

    double max_log_prob = std::max(log_prob_pos, log_prob_neg);
    double exp_log_prob_pos = std::exp(log_prob_pos - max_log_prob);
    double exp_log_prob_neg = std::exp(log_prob_neg - max_log_prob);
    double sum_exp_log_probs = exp_log_prob_pos + exp_log_prob_neg;

    result.probability = exp_log_prob_pos / sum_exp_log_probs;

    if (log_prob_pos > log_prob_neg)
    {
        result.label = vars.POS;
    }
    else
    {
        result.label = vars.NEG;
    }

    return result;
}

void NaiveBayesClassifier::predict(std::string abs_filepath_to_features, std::string abs_filepath_to_labels, bool preprocess)
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

void NaiveBayesClassifier::save(const std::string& filename) const
{
    std::ofstream outFile(filename, std::ios::binary);
    if (!outFile.is_open())
    {
        std::cerr << "Failed to open file for writing." << std::endl;
        return;
    }

    pVec->save(outFile);

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

    pVec->load(inFile);

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
