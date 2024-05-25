
#include "CountVectorizer.h"

using namespace std;

// ================================================================|
// ======================CONSTRUCTORS==============================|
// ================================================================|

CountVectorizer::CountVectorizer()
{
    binary = true;
    case_sensitive = true;
    include_stopwords = true;
}

CountVectorizer::CountVectorizer(bool binary_, bool case_sensitive_, bool include_stopwords_)
{
    binary = binary_;
    case_sensitive = case_sensitive_;
    include_stopwords = include_stopwords_;
}

CountVectorizer::~CountVectorizer()
{
}

// ===============================================================|
// ======================USER INTERFACE FUNCTIONS=================|
// ===============================================================|


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

    cout << "fitting CountVectorizer..." << endl;
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


int CountVectorizer::is_wordInSentence(Sentence sentence_, unsigned int idx)
{
    return sentence_.sentence_map.count(idx) ? 1 : 0;
}

void CountVectorizer::pushSentenceToWordArray(vector<string> new_sentence_vector)
{
    for (const string& word : new_sentence_vector)
    {
        if (!CountVectorizerContainsWord(word))
        {
            word_array.push_back(word);
            word_to_idx[word] = word_array.size() - 1;
        }
    }
}

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
            new_sentence->sentence_map[idx] = 1;
        }
    }
    if (binary)
    {
        for (auto& entry : new_sentence->sentence_map)
        {
            entry.second = 1;
        }
    }
    new_sentence->label = label_;
    return new_sentence;
}

void CountVectorizer::addSentence(string new_sentence, bool label_)
{
    vector<string> processedString;
    processedString = buildSentenceVector(new_sentence);
    pushSentenceToWordArray(processedString);
    shared_ptr<Sentence> sentObj = createSentenceObject(processedString, label_);
    sentences.push_back(sentObj);
}

bool CountVectorizer::CountVectorizerContainsWord(const string& word_to_check)
{
    return word_to_idx.count(word_to_check) > 0;
}

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

std::vector<int> CountVectorizer::getSentenceFeatures(std::vector<std::string> sentence_words) const
{
    std::vector<int> sentence_features(word_array.size(), 0);
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
            int key, value;
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
