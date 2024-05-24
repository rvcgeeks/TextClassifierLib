#ifndef NAIVEBAYESCLASSIFIER_H__
#define NAIVEBAYESCLASSIFIER_H__

#include <cmath>
#include "CountVectorizer.h"

class NaiveBayesClassifier
{
public:
    NaiveBayesClassifier();
    ~NaiveBayesClassifier();
    void shape();
    void head();
    void fit(string abs_filepath_to_features, string abs_filepath_to_labels);
    void predict(string abs_filepath_to_features, string abs_filepath_to_labels);
    int predict(string sentence);
    int totalWords();
    int totalWordsOfType(bool label_);
    float pOfType(bool label_);
    int countOccurances(string word);
    int countOccurancesOfType(string word, bool label_);
    float getWeight(vector<string> sentence, bool label_);
    void save(const std::string& filename) const;
    void load(const std::string& filename);

private:
    CountVectorizer CV;
    int total_words_of_type_true;
    float logp_true;
    int total_words_of_type_false;
    float logp_false;
    float smoothing_param_m;
    float smoothing_param_p;
};

#endif // NAIVEBAYESCLASSIFIER_H__