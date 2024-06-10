/**
 * @file CountVectorizer.cpp
 * @brief Implementation of the CountVectorizer class.
 */

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
    ifstream in;
    string feature_output;
    string label_output;
    vector<string> features;
    vector<bool> labels;

    in.open(abs_filepath_to_features);

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

    in.open(abs_filepath_to_labels);

    if (!in)
    {
        cout << "ERROR: Cannot open labels file.\n";
        return;
    }

    while (getline(in, label_output))
    {
        labels.push_back((bool)std::stoi(label_output));
    }
    in.close();

    unsigned int feature_size = features.size();
    if (feature_size != labels.size())
    {
        cout << "ERROR: Feature dimension is different from label dimension\n";
        return;
    }

    cout << "Fitting CountVectorizer..." << endl;
    int perc, prevperc;

    for (unsigned int i = 0; i < feature_size; i++)
    {
        addSentence(features[i], labels[i]);

        prevperc = perc;
        perc = int(float(i) / feature_size * 100);
        if (prevperc != perc)
        {
            cout << perc << " % done" << endl;
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
    cout << "Total unique words: " << to_string(wordArraySize) << endl;
    cout << "Documents in corpus: " << to_string(sentenceCount) << endl;
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
        for (auto sentence : sentences)
        {
            if (is_wordInSentence(*sentence, i))
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
    for (const string& word : new_sentence_vector)
    {
        if (!ContainsWord(word))
        {
            word_array.push_back(word);
            word_to_idx[word] = word_array.size() - 1;
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
shared_ptr<Sentence> CountVectorizer::createSentenceObject(vector<string> new_sentence_vector, bool label_)
{
    shared_ptr<Sentence> new_sentence(new Sentence);
    for (const auto& word : new_sentence_vector)
    {
        int idx = word_to_idx[word];
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
        for (auto& entry : new_sentence->sentence_map)
        {
            entry.second = 1.0;
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
    shared_ptr<Sentence> sentObj = createSentenceObject(processedString, label_);
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
 * @brief Split a sentence into a vector of words.
 *
 * @param sentence_ The sentence to split.
 * @return Vector of words.
 */
vector<string> CountVectorizer::buildSentenceVector(string sentence_)
{
    GlobalData vars;
    string new_word = "";
    vector<string> ret;

    for (char x : sentence_)
    {
        if (isupper(x) && !case_sensitive)
        {
            x = tolower(x);
        }
        if (x == ' ')
        {
            if (!include_stopwords && vars.stopWords.count(new_word))
            {
                new_word = "";
            }
            else
            {
                ret.push_back(new_word);
                new_word = "";
            }
        }
        else if (vars.punctuation.count(x))
        {
            ret.push_back(new_word);
            new_word = x;
            ret.push_back(new_word);
            new_word = "";
        }
        else
        {
            new_word += x;
        }
    }

    if (new_word != "")
    {
        ret.push_back(new_word);
    }

    vector<string> fixed_ret;
    for (const auto& s : ret)
    {
        if (!s.empty())
        {
            fixed_ret.push_back(s);
        }
    }
    return fixed_ret;
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
    for (const std::string& word : sentence_words)
    {
        if (word_to_idx.count(word) > 0)
        {
            int idx = word_to_idx.at(word);
            sentence_features[idx]++;
        }
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
    size_t word_array_size = word_array.size();
    outFile.write(reinterpret_cast<const char*>(&word_array_size), sizeof(word_array_size));
    for (const auto& word : word_array)
    {
        size_t word_size = word.size();
        outFile.write(reinterpret_cast<const char*>(&word_size), sizeof(word_size));
        outFile.write(word.data(), word_size);
    }

    size_t sentence_size = sentences.size();
    outFile.write(reinterpret_cast<const char*>(&sentence_size), sizeof(sentence_size));
    for (const auto& sentence : sentences)
    {
        size_t map_size = sentence->sentence_map.size();
        outFile.write(reinterpret_cast<const char*>(&map_size), sizeof(map_size));
        for (const auto& entry : sentence->sentence_map)
        {
            outFile.write(reinterpret_cast<const char*>(&entry.first), sizeof(entry.first));
            outFile.write(reinterpret_cast<const char*>(&entry.second), sizeof(entry.second));
        }
        outFile.write(reinterpret_cast<const char*>(&sentence->label), sizeof(sentence->label));
    }

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

    size_t sentence_size;
    inFile.read(reinterpret_cast<char*>(&sentence_size), sizeof(sentence_size));
    sentences.resize(sentence_size);
    for (size_t i = 0; i < sentence_size; ++i)
    {
        auto sentence = make_shared<Sentence>();
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

    inFile.read(reinterpret_cast<char*>(&binary), sizeof(binary));
    inFile.read(reinterpret_cast<char*>(&case_sensitive), sizeof(case_sensitive));
    inFile.read(reinterpret_cast<char*>(&include_stopwords), sizeof(include_stopwords));
}
