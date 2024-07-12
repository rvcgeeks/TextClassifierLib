#ifndef TFIDFVECTORIZER_H__
#define TFIDFVECTORIZER_H__

#include <string>
#include <vector>
#include <memory>
#include <iostream>
#include <unordered_map>
#include <fstream>
#include <sstream>

#include "GlobalData.h"
#include "BaseVectorizer.h"

/**
 * @file TfidfVectorizer.h
 * @brief Definition of TfidfVectorizer class.
 */

/**
 * @class TfidfVectorizer
 * @brief A class for TF-IDF vectorization.
 * 
 * TF-IDF (Term Frequency-Inverse Document Frequency) is a numerical statistic that reflects 
 * the importance of a word in a document relative to a collection of documents. It is widely 
 * used in information retrieval and text mining.
 * 
 * The TF-IDF value for a term in a document is calculated as follows:
 * \f[
 * \text{tfidf}(t, d, D) = \text{tf}(t, d) \times \text{idf}(t, D)
 * \f]
 * where:
 * - \f$t\f$ is the term (word),
 * - \f$d\f$ is the document (sentence),
 * - \f$D\f$ is the collection of all documents (sentences),
 * - \f$\text{tf}(t, d)\f$ is the term frequency of \f$t\f$ in \f$d\f$,
 * - \f$\text{idf}(t, D)\f$ is the inverse document frequency of \f$t\f$ in \f$D\f$.
 */
class TfidfVectorizer : public BaseVectorizer
{
public:

    /**
     * @brief Default constructor for TfidfVectorizer.
     */
    TfidfVectorizer();

    /**
     * @brief Parameterized constructor for TfidfVectorizer.
     * @param binary_ Whether to use binary representation.
     * @param case_sensitive_ Whether to consider case sensitivity.
     * @param include_stopwords_ Whether to include stopwords.
     */
    TfidfVectorizer(bool binary_, bool case_sensitive_, bool include_stopwords_);

    /**
     * @brief Destructor for TfidfVectorizer.
     */
    ~TfidfVectorizer();

    /**
     * @brief Fit the vectorizer on provided features and labels.
     * @param abs_filepath_to_features Absolute filepath to features file.
     * @param abs_filepath_to_labels Absolute filepath to labels file.
     */
    void fit(std::string abs_filepath_to_features, std::string abs_filepath_to_labels);

    /**
     * @brief Print the shape of the data.
     */
    void shape();

    /**
     * @brief Print the first few elements of the data.
     */
    void head();

    /**
     * @brief Check if a word is present in a sentence.
     * @param sentence_ The sentence to check.
     * @param idx Index of the word.
     * @return 1 if the word is present, 0 otherwise.
     */
    int is_wordInSentence(Sentence sentence_, unsigned int idx);

    /**
     * @brief Add a sentence to the word array.
     * @param new_sentence_vector The sentence to add.
     */
    void pushSentenceToWordArray(std::vector<std::string> new_sentence_vector);

    /**
     * @brief Create a sentence object.
     * @param new_sentence_vector The sentence vector.
     * @param label_ The label of the sentence.
     * @return Shared pointer to the created sentence object.
     */
    std::tr1::shared_ptr<Sentence> createSentenceObject(std::vector<std::string> new_sentence_vector, bool label_);

    /**
     * @brief Add a sentence to the vectorizer.
     * @param new_sentence The sentence to add.
     * @param label_ The label of the sentence.
     */
    void addSentence(std::string new_sentence, bool label_);

    /**
     * @brief Check if a word is present in the vectorizer.
     * @param word_to_check The word to check.
     * @return True if the word is present, false otherwise.
     */
    bool ContainsWord(const std::string& word_to_check);

    /**
     * @brief Get the word at a given index.
     * @param idx The index of the word.
     * @return The word at the specified index.
     */
    std::string getWord(int idx) { return word_array[idx]; }

    /**
     * @brief Get the sentence at a given index.
     * @param idx The index of the sentence.
     * @return Shared pointer to the sentence at the specified index.
     */
    std::tr1::shared_ptr<Sentence> getSentence(int idx) { return sentences[idx]; }

    /**
     * @brief Get the size of the word array.
     * @return The size of the word array.
     */
    unsigned int getWordArraySize() { return word_array.size(); }

    /**
     * @brief Get the count of sentences.
     * @return The count of sentences.
     */
    unsigned int getSentenceCount() { return sentences.size(); }
    
    /**
     * @brief Get the TF-IDF features for a sentence.
     * @param sentence_words The words in the sentence.
     * @return The TF-IDF features for the sentence.
     */
    std::vector<double> getSentenceFeatures(std::vector<std::string> sentence_words) const;

    std::vector<double> getFrequencies(std::tr1::unordered_map<int, double> term_freqs) const;

    /**
     * @brief Save the vectorizer to a file.
     * @param outFile The output file stream.
     */
    void save(std::ofstream& outFile) const;

    /**
     * @brief Load the vectorizer from a file.
     * @param inFile The input file stream.
     */
    void load(std::ifstream& inFile);

private:
    std::tr1::unordered_map<int, double> idf_values;  ///< Store IDF values
};

#endif // TFIDFVECTORIZER_H__
