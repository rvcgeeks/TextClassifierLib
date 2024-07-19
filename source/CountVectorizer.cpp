/**
 * @file CountVectorizer.cpp
 * @brief Implementation of the CountVectorizer class.
 */

/*++

Revision History:
	Date:	Jun 28, 2024.
	Author:	Rajas Chavadekar.
	Desc:	Created.

--*/

#include "CountVectorizer.h"

using namespace std;

/**
 * @brief Default constructor.
 *
 * Defaults to binary=true, case_sensitive=true, and include_stopwords=true.
 */
CountVectorizer::CountVectorizer()
{
    binary = true;
    case_sensitive = true;
    include_stopwords = true;
    this_vectorizer_id = ID_VECTORIZER_COUNT;
}

/**
 * @brief Constructor with options.
 *
 * @param binary_ Boolean flag indicating if binary vectors are used.
 * @param case_sensitive_ Boolean flag indicating if case sensitivity is considered.
 * @param include_stopwords_ Boolean flag indicating if stop words are included.
 */
CountVectorizer::CountVectorizer(bool binary_, bool case_sensitive_, bool include_stopwords_)
{
    binary = binary_;
    case_sensitive = case_sensitive_;
    include_stopwords = include_stopwords_;
}

/**
 * @brief Destructor.
 */
CountVectorizer::~CountVectorizer()
{
}

/**
 * @brief Fit the vectorizer on the given dataset.
 *
 * @param abs_filepath_to_features Absolute file path to the features file.
 * @param abs_filepath_to_labels Absolute file path to the labels file.
 */
void CountVectorizer::fit(string abs_filepath_to_features, string abs_filepath_to_labels)
{
    ifstream in, in1;
    string feature_output;
    string label_output;
    vector<string> features;
    vector<bool> labels;

    in.open(abs_filepath_to_features.c_str());

    if (!in)
    {
        cout << "ERROR: Cannot open features file.\n";
        return;
    }

    while (getline(in, feature_output))
    {
        features.push_back(feature_output);
    }
    in.close();

    in1.open(abs_filepath_to_labels.c_str());

    if (!in1)
    {
        cout << "ERROR: Cannot open labels file.\n";
        return;
    }

    while (getline(in1, label_output))
    {
		std::istringstream iss(label_output);
		int label;
		iss >> label;
        labels.push_back(label);
    }
    in1.close();

    unsigned int feature_size = features.size();
    if (feature_size != labels.size())
    {
        cout << "ERROR: Feature dimension is different from label dimension\n";
        return;
    }

    cout << "Fitting CountVectorizer..." << endl;
    int perc = 0, prevperc = -1;

    for (unsigned int i = 0; i < feature_size; i++)
    {
        addSentence(features[i], labels[i]);

        perc = int(float(i) / feature_size * 100);
        if (prevperc != perc)
        {
            cout << perc << " % done" << endl;
            prevperc = perc;
        }
    }
    cout << endl;
}

/**
 * @brief Print the dimensions of the CountVectorizer object.
 */
void CountVectorizer::shape()
{
    unsigned int wordArraySize = getWordArraySize();
    unsigned int sentenceCount = getSentenceCount();
    cout << "------------------------------" << endl;
    cout << "Current CountVectorizer Shape:" << endl;
    cout << "Total unique words: " << wordArraySize << endl;
    cout << "Documents in corpus: " << sentenceCount << endl;
    cout << "------------------------------" << endl;
}

/**
 * @brief Print a dictionary-like representation of the CountVectorizer object (first 10).
 */
void CountVectorizer::head()
{
    int count = 0;
    unsigned int wordArraySize = getWordArraySize();
    if (wordArraySize > 10)
    {
        wordArraySize = 10;
    }
    cout << "------------------------------" << endl;
    cout << "Current CountVectorizer Head:" << endl;
    for (unsigned int i = 0; i < wordArraySize; i++)
    {
        for (vector< std::tr1::shared_ptr<Sentence> >::const_iterator it = sentences.begin(); it != sentences.end(); ++it)
        {
            if (is_wordInSentence(**it, i))
            {
                count++;
            }
        }
        cout << getWord(i) << ": " << count << endl;
        count = 0;
    }
    cout << "------------------------------" << endl;
}

// ===========================================================|
// ======================HELPERS==============================|
// ===========================================================|

/**
 * @brief Check if a word is in the sentence.
 *
 * @param sentence_ The sentence to check.
 * @param idx The index of the word to check.
 * @return Integer casted boolean indicating presence of the word.
 */
int CountVectorizer::is_wordInSentence(Sentence sentence_, unsigned int idx)
{
    return sentence_.sentence_map.count(idx) ? 1 : 0;
}

/**
 * @brief Update the word array with newly discovered words from a sentence.
 *
 * @param new_sentence_vector The sentence vector containing new words.
 */
void CountVectorizer::pushSentenceToWordArray(vector<string> new_sentence_vector)
{
    for (vector<string>::const_iterator it = new_sentence_vector.begin(); it != new_sentence_vector.end(); ++it)
    {
        if (!ContainsWord(*it) && !histogram.count(*it))
        {
            word_array.push_back(*it);
            word_to_idx[*it] = word_array.size() - 1;
        }
    }
}

/**
 * @brief Create a Sentence object from a vector of words.
 *
 * @param new_sentence_vector The vector of words forming the sentence.
 * @param label_ Boolean label for the sentence.
 * @return Shared pointer to the created Sentence object.
 */
std::tr1::shared_ptr<Sentence> CountVectorizer::createSentenceObject(vector<string> new_sentence_vector, bool label_)
{
    std::tr1::shared_ptr<Sentence> new_sentence(new Sentence);
    for (vector<string>::const_iterator it = new_sentence_vector.begin(); it != new_sentence_vector.end(); ++it)
    {
        if (histogram.count(*it))
        {
            continue;
        }

        int idx = word_to_idx[*it];
        if (new_sentence->sentence_map.count(idx))
        {
            new_sentence->sentence_map[idx]++;
        }
        else
        {
            new_sentence->sentence_map[idx] = 1.0;
        }
    }
    if (binary)
    {
        for (tr1::unordered_map<int, double>::iterator it = new_sentence->sentence_map.begin(); it != new_sentence->sentence_map.end(); ++it)
        {
            it->second = 1.0;
        }
    }
    new_sentence->label = label_;
    return new_sentence;
}

/**
 * @brief Add a sentence to the CountVectorizer.
 *
 * @param new_sentence The new sentence to add.
 * @param label_ Boolean label for the sentence.
 */
void CountVectorizer::addSentence(string new_sentence, bool label_)
{
    vector<string> processedString;
    processedString = buildSentenceVector(new_sentence);
    pushSentenceToWordArray(processedString);
    std::tr1::shared_ptr<Sentence> sentObj = createSentenceObject(processedString, label_);
    sentences.push_back(sentObj);
}

/**
 * @brief Check if the CountVectorizer already contains the word.
 *
 * @param word_to_check The word to check.
 * @return Boolean indicating if the word is present.
 */
bool CountVectorizer::ContainsWord(const string& word_to_check)
{
    return word_to_idx.count(word_to_check) > 0;
}

/**
 * @brief Get the feature vector for a given sentence.
 *
 * @param sentence_words The words of the sentence.
 * @return Vector of feature values.
 */
std::vector<double> CountVectorizer::getSentenceFeatures(std::vector<std::string> sentence_words) const
{
    std::vector<double> sentence_features(word_array.size(), 0.0);
    for (std::vector<std::string>::const_iterator it = sentence_words.begin(); it != sentence_words.end(); ++it)
    {
        if (word_to_idx.count(*it) > 0)
        {
            int idx = getmapval(word_to_idx, *it);
            sentence_features[idx]++;
        }
    }
    return sentence_features;
}

std::vector<double> CountVectorizer::getFrequencies(std::tr1::unordered_map<int, double> term_freqs) const
{
    std::vector<double> sentence_features(word_array.size(), 0.0);
    for (std::tr1::unordered_map<int, double>::const_iterator it = term_freqs.begin(); it != term_freqs.end(); ++it)
    {
        int term_idx = it->first;
        int term_freq = it->second;
        double tf = term_freq;
        sentence_features[term_idx] = tf;
    }
    return sentence_features;
}

/**
 * @brief Save the CountVectorizer model to a file.
 *
 * @param outFile Output file stream to save the model.
 */
void CountVectorizer::save(std::ofstream& outFile) const
{
	outFile.write(reinterpret_cast<const char*>(&vers_info), sizeof(vers_info));

    size_t word_array_size = word_array.size();
    outFile.write(reinterpret_cast<const char*>(&word_array_size), sizeof(word_array_size));
    for (std::vector<std::string>::const_iterator it = word_array.begin(); it != word_array.end(); ++it)
    {
        size_t word_size = it->size();
        outFile.write(reinterpret_cast<const char*>(&word_size), sizeof(word_size));
        outFile.write(it->data(), word_size);
    }

    /*
    size_t sentence_size = sentences.size();
    outFile.write(reinterpret_cast<const char*>(&sentence_size), sizeof(sentence_size));
    for (std::vector< std::tr1::shared_ptr<Sentence> >::const_iterator it = sentences.begin(); it != sentences.end(); ++it)
    {
        size_t map_size = (*it)->sentence_map.size();
        outFile.write(reinterpret_cast<const char*>(&map_size), sizeof(map_size));
        for (tr1::unordered_map<int, double>::const_iterator map_it = (*it)->sentence_map.begin(); map_it != (*it)->sentence_map.end(); ++map_it)
        {
            outFile.write(reinterpret_cast<const char*>(&map_it->first), sizeof(map_it->first));
            outFile.write(reinterpret_cast<const char*>(&map_it->second), sizeof(map_it->second));
        }
        outFile.write(reinterpret_cast<const char*>(&(*it)->label), sizeof((*it)->label));
    }
    */

    outFile.write(reinterpret_cast<const char*>(&binary), sizeof(binary));
    outFile.write(reinterpret_cast<const char*>(&case_sensitive), sizeof(case_sensitive));
    outFile.write(reinterpret_cast<const char*>(&include_stopwords), sizeof(include_stopwords));
}

/**
 * @brief Load the CountVectorizer model from a file.
 *
 * @param inFile Input file stream to load the model.
 */
void CountVectorizer::load(std::ifstream& inFile)
{
    word_array.clear();
    word_to_idx.clear();
    sentences.clear();

	inFile.read(reinterpret_cast<char*>(&vers_info), sizeof(vers_info));

    size_t word_array_size;
    inFile.read(reinterpret_cast<char*>(&word_array_size), sizeof(word_array_size));
    word_array.resize(word_array_size);
    for (size_t i = 0; i < word_array_size; ++i)
    {
        size_t word_size;
        inFile.read(reinterpret_cast<char*>(&word_size), sizeof(word_size));
        word_array[i].resize(word_size);
        inFile.read(&word_array[i][0], word_size);
        word_to_idx[word_array[i]] = i;
    }

    /*
    size_t sentence_size;
    inFile.read(reinterpret_cast<char*>(&sentence_size), sizeof(sentence_size));
    sentences.resize(sentence_size);
    for (size_t i = 0; i < sentence_size; ++i)
    {
        std::tr1::shared_ptr<Sentence> sentence(new Sentence);
        size_t map_size;
        inFile.read(reinterpret_cast<char*>(&map_size), sizeof(map_size));
        for (size_t j = 0; j < map_size; ++j)
        {
            int key;
            double value;
            inFile.read(reinterpret_cast<char*>(&key), sizeof(key));
            inFile.read(reinterpret_cast<char*>(&value), sizeof(value));
            sentence->sentence_map[key] = value;
        }
        inFile.read(reinterpret_cast<char*>(&sentence->label), sizeof(sentence->label));
        sentences[i] = sentence;
    }
    */

    inFile.read(reinterpret_cast<char*>(&binary), sizeof(binary));
    inFile.read(reinterpret_cast<char*>(&case_sensitive), sizeof(case_sensitive));
    inFile.read(reinterpret_cast<char*>(&include_stopwords), sizeof(include_stopwords));
}
