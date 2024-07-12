
#include "TfidfVectorizer.h"

#include <cmath>

int stoi(const std::string& str) {
    std::istringstream iss(str);
    int result;
    iss >> result;
    return result;
}

using namespace std;

double log1p(double x)
{
	return log(1.0 + x);
}

// ================================================================|
// ======================CONSTRUCTORS==============================|
// ================================================================|

TfidfVectorizer::TfidfVectorizer()
{
    binary = true;
    case_sensitive = true;
    include_stopwords = true;
    this_vectorizer_id = ID_VECTORIZER_TFIDF;
}

TfidfVectorizer::TfidfVectorizer(bool binary_, bool case_sensitive_, bool include_stopwords_)
{
    binary = binary_;
    case_sensitive = case_sensitive_;
    include_stopwords = include_stopwords_;
}

TfidfVectorizer::~TfidfVectorizer()
{
}

// ===============================================================|
// ======================USER INTERFACE FUNCTIONS=================|
// ===============================================================|

void TfidfVectorizer::fit(string abs_filepath_to_features, string abs_filepath_to_labels)
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

    cout << "fitting TfidfVectorizer..." << endl;
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

    // Calculate IDF values
    for (tr1::unordered_map<string, int>::const_iterator word_idx = word_to_idx.begin(); word_idx != word_to_idx.end(); ++word_idx)
    {
        int doc_count = 0;
        for (vector<tr1::shared_ptr<Sentence> >::const_iterator sentence = sentences.begin(); sentence != sentences.end(); ++sentence)
        {
            if ((*sentence)->sentence_map.count(word_idx->second))
            {
                doc_count++;
            }
        }
        idf_values[word_idx->second] = log1p(double(sentences.size()) / (1 + doc_count));
    }
}

void TfidfVectorizer::shape()
{
    unsigned int wordArraySize = getWordArraySize();
    unsigned int sentenceCount = getSentenceCount();
    cout << "------------------------------" << endl;
    cout << "Current TfidfVectorizer Shape:" << endl;
    cout << "Total unique words: " << wordArraySize << endl;
    cout << "Documents in corpus: " << sentenceCount << endl;
    cout << "------------------------------" << endl;
}

void TfidfVectorizer::head()
{
    int count = 0;
    unsigned int wordArraySize = getWordArraySize();
    if (wordArraySize > 10)
    {
        wordArraySize = 10;
    }
    cout << "------------------------------" << endl;
    cout << "Current TfidfVectorizer Head:" << endl;
    for (unsigned int i = 0; i < wordArraySize; i++)
    {
        for (vector<tr1::shared_ptr<Sentence> >::const_iterator sentence = sentences.begin(); sentence != sentences.end(); ++sentence)
        {
            if (is_wordInSentence(**sentence, i))
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

int TfidfVectorizer::is_wordInSentence(Sentence sentence_, unsigned int idx)
{
    return sentence_.sentence_map.count(idx) ? 1 : 0;
}

void TfidfVectorizer::pushSentenceToWordArray(vector<string> new_sentence_vector)
{
    for (vector<string>::const_iterator word = new_sentence_vector.begin(); word != new_sentence_vector.end(); ++word)
    {
        if (!ContainsWord(*word) && !histogram.count(*word))
        {
            word_array.push_back(*word);
            word_to_idx[*word] = word_array.size() - 1;
        }
    }
}

tr1::shared_ptr<Sentence> TfidfVectorizer::createSentenceObject(vector<string> new_sentence_vector, bool label_)
{
    tr1::shared_ptr<Sentence> new_sentence(new Sentence);
    tr1::unordered_map<int, int> term_freqs;

    for (vector<string>::const_iterator word = new_sentence_vector.begin(); word != new_sentence_vector.end(); ++word)
    {
        if (histogram.count(*word))
        {
            continue;
        }

        int idx = word_to_idx[*word];
        if (term_freqs.count(idx))
        {
            term_freqs[idx]++;
        }
        else
        {
            term_freqs[idx] = 1;
        }
    }

    for (tr1::unordered_map<int, int>::const_iterator entry = term_freqs.begin(); entry != term_freqs.end(); ++entry)
    {
        new_sentence->sentence_map[entry->first] = entry->second;
    }

    new_sentence->label = label_;
    return new_sentence;
}

void TfidfVectorizer::addSentence(string new_sentence, bool label_)
{
    vector<string> processedString;
    processedString = buildSentenceVector(new_sentence);
    pushSentenceToWordArray(processedString);
    tr1::shared_ptr<Sentence> sentObj = createSentenceObject(processedString, label_);
    sentences.push_back(sentObj);
}

bool TfidfVectorizer::ContainsWord(const string& word_to_check)
{
    return word_to_idx.count(word_to_check) > 0;
}

vector<double> TfidfVectorizer::getFrequencies(tr1::unordered_map<int, double> term_freqs) const
{
    vector<double> sentence_features(word_array.size(), 0.0);
    for (tr1::unordered_map<int, double>::const_iterator entry = term_freqs.begin(); entry != term_freqs.end(); ++entry)
    {
        int term_idx = entry->first;
        int term_freq = entry->second;
        double tf = term_freq;
        double idf = getmapval(idf_values, term_idx);
        sentence_features[term_idx] = tf * idf;
    }
    return sentence_features;
}

vector<double> TfidfVectorizer::getSentenceFeatures(vector<string> sentence_words) const
{
    vector<double> sentence_features(word_array.size(), 0.0);
    tr1::unordered_map<int, int> term_freqs;

    for (vector<string>::const_iterator word = sentence_words.begin(); word != sentence_words.end(); ++word)
    {
        if (word_to_idx.count(*word) > 0)
        {
            int idx = getmapval(word_to_idx, *word);
            term_freqs[idx]++;
        }
    }

    for (tr1::unordered_map<int, int>::const_iterator entry = term_freqs.begin(); entry != term_freqs.end(); ++entry)
    {
        int term_idx = entry->first;
        int term_freq = entry->second;
        double tf = term_freq;
        double idf = getmapval(idf_values, term_idx);
        sentence_features[term_idx] = tf * idf;
    }
    return sentence_features;
}

void TfidfVectorizer::save(ofstream& outFile) const
{
    size_t word_array_size = word_array.size();
    outFile.write(reinterpret_cast<const char*>(&word_array_size), sizeof(word_array_size));
    for (vector<string>::const_iterator word = word_array.begin(); word != word_array.end(); ++word)
    {
        size_t word_size = word->size();
        outFile.write(reinterpret_cast<const char*>(&word_size), sizeof(word_size));
        outFile.write(word->data(), word_size);
    }

    size_t idf_size = idf_values.size();
    outFile.write(reinterpret_cast<const char*>(&idf_size), sizeof(idf_size));
    for (tr1::unordered_map<int, double>::const_iterator entry = idf_values.begin(); entry != idf_values.end(); ++entry)
    {
        outFile.write(reinterpret_cast<const char*>(&entry->first), sizeof(entry->first));
        outFile.write(reinterpret_cast<const char*>(&entry->second), sizeof(entry->second));
    }

    outFile.write(reinterpret_cast<const char*>(&binary), sizeof(binary));
    outFile.write(reinterpret_cast<const char*>(&case_sensitive), sizeof(case_sensitive));
    outFile.write(reinterpret_cast<const char*>(&include_stopwords), sizeof(include_stopwords));
}

void TfidfVectorizer::load(ifstream& inFile)
{
    word_array.clear();
    word_to_idx.clear();
    sentences.clear();
    idf_values.clear();

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

    size_t idf_size;
    inFile.read(reinterpret_cast<char*>(&idf_size), sizeof(idf_size));
    for (size_t i = 0; i < idf_size; ++i)
    {
        int key;
        double value;
        inFile.read(reinterpret_cast<char*>(&key), sizeof(key));
        inFile.read(reinterpret_cast<char*>(&value), sizeof(value));
        idf_values[key] = value;
    }

    inFile.read(reinterpret_cast<char*>(&binary), sizeof(binary));
    inFile.read(reinterpret_cast<char*>(&case_sensitive), sizeof(case_sensitive));
    inFile.read(reinterpret_cast<char*>(&include_stopwords), sizeof(include_stopwords));
}
