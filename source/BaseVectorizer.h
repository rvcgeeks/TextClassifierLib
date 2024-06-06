
#ifndef BASEVECTORIZER_H__
#define BASEVECTORIZER_H__

#include <string>
#include <vector>
#include <unordered_map>
#include <memory>
#include <fstream>

#include "GlobalData.h"

using namespace std;

#define ID_VECTORIZER_COUNT     1
#define ID_VECTORIZER_TFIDF     2

struct Sentence
{
    unordered_map<int, double> sentence_map;
    bool label;
};

class BaseVectorizer
{
public:
    virtual ~BaseVectorizer() = default;

    virtual void fit(string abs_filepath_to_features, string abs_filepath_to_labels) = 0;
    virtual void shape() = 0;
    virtual void head() = 0;

    void setBinary(bool bool_) { binary = bool_; }
    void setCaseSensitive(bool bool_) { case_sensitive = bool_; }
    void setIncludeStopWords(bool bool_) { include_stopwords = bool_; }

    virtual void addSentence(string new_sentence, bool label_) = 0;
    virtual bool ContainsWord(const string& word_to_check) = 0;
    virtual vector<string> buildSentenceVector(string sentence_) = 0;
    virtual std::vector<double> getSentenceFeatures(std::vector<std::string> sentence_words) const = 0;

    string getWord(int idx) { return word_array[idx]; }
    shared_ptr<Sentence> getSentence(int idx) { return sentences[idx]; }
    unsigned int getWordArraySize() { return word_array.size(); }
    unsigned int getSentenceCount() { return sentences.size(); }
    
    virtual void save(ofstream& outFile) const = 0;
    virtual void load(ifstream& inFile) = 0;

    friend class NaiveBayesClassifier;
    friend class LogisticRegressionClassifier;
    friend class SVCClassifier;
    friend class KNNClassifier;
    friend class RandomForestClassifier;
    friend class GradientBoostingClassifier;

protected:
    vector<string> word_array;
    unordered_map<string, int> word_to_idx;
    vector<shared_ptr<Sentence>> sentences;
    bool binary;
    bool case_sensitive;
    bool include_stopwords;
};

#endif // BASEVECTORIZER_H__
