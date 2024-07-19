/**
 * @file CountVectorizer.h
 * @brief Header file for the CountVectorizer class.
 */

/*++

Revision History:
	Date:	Jun 28, 2024.
	Author:	Rajas Chavadekar.
	Desc:	Created.

--*/

#ifndef COUNTERVECTORIZER_H__
#define COUNTERVECTORIZER_H__

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

/**
 * @class CountVectorizer
 * @brief Converts a collection of text documents to a matrix of token counts.
 *
 * This class provides functionality to convert text documents into vectors
 * of token counts, where each token is a feature. It can be configured to
 * consider case sensitivity and stop words.
 *
 * The formula used for count vectorization is:
 * \f[
 * \text{count}(w, d) = \sum_{t \in d} \delta(w, t)
 * \f]
 * where \(\delta(w, t)\) is 1 if \(w = t\) and 0 otherwise.
 *
 * CountVectorizer can be useful in scenarios where the frequency of words
 * in a document is an important feature for classification tasks.
 */
class CountVectorizer : public BaseVectorizer
{
public:
    // ======================CONSTRUCTORS==============================|

    /**
     * @brief Default constructor.
     *
     * Defaults to case_sensitive=true and include_stopwords=true.
     */
    CountVectorizer();

    /**
     * @brief Constructor with options.
     *
     * @param binary_ Boolean flag indicating if binary vectors are used.
     * @param case_sensitive_ Boolean flag indicating if case sensitivity is considered.
     * @param include_stopwords_ Boolean flag indicating if stop words are included.
     */
    CountVectorizer(bool binary_, bool case_sensitive_, bool include_stopwords_);

    /**
     * @brief Destructor.
     */
    ~CountVectorizer();

    // ======================USER INTERFACE FUNCTIONS==================|

    /**
     * @brief Fit the vectorizer on the given dataset.
     *
     * @param abs_filepath_to_features Absolute file path to the features file.
     * @param abs_filepath_to_labels Absolute file path to the labels file.
     */
    void fit(string abs_filepath_to_features, string abs_filepath_to_labels) override;

    /**
     * @brief Print the dimensions of the CountVectorizer object.
     */
    void shape() override;

    /**
     * @brief Print a dictionary-like representation of the CountVectorizer object (first 10).
     */
    void head() override;

    // ======================HELPERS===================================|

    /**
     * @brief Check if a word is in the sentence.
     *
     * @param sentence_ The sentence to check.
     * @param idx The index of the word to check.
     * @return Integer casted boolean indicating presence of the word.
     */
    int is_wordInSentence(Sentence sentence_, unsigned int idx);

    /**
     * @brief Update the word array with newly discovered words from a sentence.
     *
     * @param new_sentence_vector The sentence vector containing new words.
     */
    void pushSentenceToWordArray(vector<string> new_sentence_vector);

    /**
     * @brief Create a Sentence object from a vector of words.
     *
     * @param new_sentence_vector The vector of words forming the sentence.
     * @param label_ Boolean label for the sentence.
     * @return Shared pointer to the created Sentence object.
     */
    shared_ptr<Sentence> createSentenceObject(vector<string> new_sentence_vector, bool label_);

    /**
     * @brief Add a sentence to the CountVectorizer.
     *
     * @param new_sentence The new sentence to add.
     * @param label_ Boolean label for the sentence.
     */
    void addSentence(string new_sentence, bool label_) override;

    /**
     * @brief Check if the CountVectorizer already contains the word.
     *
     * @param word_to_check The word to check.
     * @return Boolean indicating if the word is present.
     */
    bool ContainsWord(const string& word_to_check) override;

    /**
     * @brief Get the feature vector for a given sentence.
     *
     * @param sentence_words The words of the sentence.
     * @return Vector of feature values.
     */
    std::vector<double> getSentenceFeatures(std::vector<std::string> sentence_words) const override;

    std::vector<double> getFrequencies(std::unordered_map<int, double> term_freqs) const override;

    /**
     * @brief Save the CountVectorizer model to a file.
     *
     * @param outFile Output file stream to save the model.
     */
    void save(std::ofstream& outFile) const override;

    /**
     * @brief Load the CountVectorizer model from a file.
     *
     * @param inFile Input file stream to load the model.
     */
    void load(std::ifstream& inFile) override;
};

#endif // COUNTERVECTORIZER_H__
