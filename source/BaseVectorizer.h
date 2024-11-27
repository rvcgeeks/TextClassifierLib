
/**
 * @file BaseVectorizer.h
 * @brief Declaration of the BaseVectorizer class and related structures.
 */

/*++

Revision History:
	Date:	Jun 28, 2024.
	Author:	Rajas Chavadekar.
	Desc:	Created.

--*/

#ifndef BASEVECTORIZER_H__
#define BASEVECTORIZER_H__

#include <string>
#include <vector>
#include <unordered_map>
#include <memory>
#include <fstream>
#include <cstring>

#include "GlobalData.h"

#define ID_VECTORIZER_COUNT     1
#define ID_VECTORIZER_TFIDF     2

#define VERSION_INFO_SIZE		32

/**
 * @brief Structure representing a sentence with its corresponding label.
 */
struct Sentence
{
    std::unordered_map<int, double> sentence_map; /**< Map representing the sentence. */
    bool label; /**< Label indicating the classification of the sentence. */
};

/**
 * @brief Abstract class defining the interface for vectorizers.
 */
class BaseVectorizer
{
public:
    virtual ~BaseVectorizer() = default;

    /**
     * @brief Fits the vectorizer on the provided features and labels data.
     * 
     * @param abs_filepath_to_features Absolute file path to the features data.
     * @param abs_filepath_to_labels Absolute file path to the labels data.
     */
    virtual void fit(std::string abs_filepath_to_features, std::string abs_filepath_to_labels) = 0;

    void scanForSparseHistogram(std::string abs_filepath_to_features, int minfrequency);

    /**
     * @brief Returns the shape of the vectorized data.
     */
    virtual void shape() = 0;

    /**
     * @brief Displays the first few elements of the vectorized data.
     */
    virtual void head() = 0;

    /**
     * @brief Sets whether to use binary encoding for vectorization.
     * 
     * @param bool_ Boolean value indicating binary encoding.
     */
    void setBinary(bool bool_) { binary = bool_; }

    /**
     * @brief Sets whether the vectorizer is case sensitive.
     * 
     * @param bool_ Boolean value indicating case sensitivity.
     */
    void setCaseSensitive(bool bool_) { case_sensitive = bool_; }

    /**
     * @brief Sets whether to include stop words in the vectorization process.
     * 
     * @param bool_ Boolean value indicating inclusion of stop words.
     */
    void setIncludeStopWords(bool bool_) { include_stopwords = bool_; }

    /**
     * @brief Adds a new sentence to the vectorizer.
     * 
     * @param new_sentence The new sentence to be added.
     * @param label_ The label associated with the new sentence.
     */
    virtual void addSentence(std::string new_sentence, bool label_) = 0;

    /**
     * @brief Checks if a word is present in the vectorizer.
     * 
     * @param word_to_check The word to check for.
     * @return True if the word is present, false otherwise.
     */
    virtual bool ContainsWord(const std::string& word_to_check) = 0;

    /**
     * @brief Builds a vector representation of a sentence.
     * 
     * @param sentence_ The sentence to be vectorized.
     * @return Vector representation of the sentence.
     */
    std::vector<std::string> buildSentenceVector(std::string sentence_, bool preprocess=false);

    /**
     * @brief Retrieves the feature vector of a sentence.
     * 
     * @param sentence_words Vector representation of the sentence.
     * @return Feature vector of the sentence.
     */
    virtual std::vector<double> getSentenceFeatures(std::vector<std::string> sentence_words) const = 0;

    virtual std::vector<double> getFrequencies(std::unordered_map<int, double> term_freqs) const = 0;

    void setVersionInfo(char* vers_info_in);

    /**
     * @brief Retrieves the word at the specified index.
     * 
     * @param idx The index of the word.
     * @return The word at the specified index.
     */
    std::string getWord(int idx) { return word_array[idx]; }

    /**
     * @brief Retrieves the sentence at the specified index.
     * 
     * @param idx The index of the sentence.
     * @return Shared pointer to the sentence.
     */
    std::shared_ptr<Sentence> getSentence(int idx) { return sentences[idx]; }

    /**
     * @brief Retrieves the size of the word array.
     * 
     * @return Size of the word array.
     */
    unsigned int getWordArraySize() { return word_array.size(); }

    /**
     * @brief Retrieves the count of sentences.
     * 
     * @return Count of sentences.
     */
    unsigned int getSentenceCount() { return sentences.size(); }
    
    /**
     * @brief Saves the vectorizer to a file.
     * 
     * @param outFile Output file stream.
     */
    virtual void save(std::ofstream& outFile) const = 0;

    /**
     * @brief Loads the vectorizer from a file.
     * 
     * @param inFile Input file stream.
     */
    virtual void load(std::ifstream& inFile) = 0;

    friend class NaiveBayesClassifier;
    friend class LogisticRegressionClassifier;
    friend class SVCClassifier;
    friend class KNNClassifier;
    friend class RandomForestClassifier;
    friend class GradientBoostingClassifier;

protected:
    std::vector<std::string> word_array; /**< Array storing words. */
    std::unordered_map<std::string, int> word_to_idx; /**< Map of words to their indices. */
    std::vector<std::shared_ptr<Sentence>> sentences; /**< Vector storing sentences. */
    std::unordered_map<std::string, int> histogram;
    int this_vectorizer_id;
    bool binary; /**< Flag indicating binary encoding. */
    bool case_sensitive; /**< Flag indicating case sensitivity. */
    bool include_stopwords; /**< Flag indicating inclusion of stop words. */
    char vers_info[VERSION_INFO_SIZE];
};

#endif // BASEVECTORIZER_H__
