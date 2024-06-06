#ifndef TFIDFVECTORIZER_H__
#define TFIDFVECTORIZER_H__

#include <string>
#include <vector>
#include <memory>
#include <iostream>
#include <unordered_set>
#include <unordered_map>
#include <fstream>
#include <sstream>

#include "GlobalData.h"
#include "BaseVectorizer.h"

class TfidfVectorizer : public BaseVectorizer
{
public:
    // ======================CONSTRUCTORS==============================|

    TfidfVectorizer();
    TfidfVectorizer(bool binary_, bool case_sensitive_, bool include_stopwords_);

    ~TfidfVectorizer();

    // ======================USER INTERFACE FUNCTIONS==================|

    void fit(string abs_filepath_to_features, string abs_filepath_to_labels) override;
    void shape() override;
    void head() override;

    // ======================HELPERS===================================|

    int is_wordInSentence(Sentence sentence_, unsigned int idx);
    void pushSentenceToWordArray(vector<string> new_sentence_vector);
    shared_ptr<Sentence> createSentenceObject(vector<string> new_sentence_vector, bool label_);
    void addSentence(string new_sentence, bool label_) override;
    bool ContainsWord(const string& word_to_check) override;
    vector<string> buildSentenceVector(string sentence_) override;

    string getWord(int idx) { return word_array[idx]; }
    shared_ptr<Sentence> getSentence(int idx) { return sentences[idx]; }
    unsigned int getWordArraySize() { return word_array.size(); }
    unsigned int getSentenceCount() { return sentences.size(); }
    
    std::vector<double> getSentenceFeatures(std::vector<std::string> sentence_words) const override;

    void save(std::ofstream& outFile) const override;
    void load(std::ifstream& inFile) override;

private:
    unordered_map<int, double> idf_values;  // Store IDF values
};

#endif // TFIDFVECTORIZER_H__
