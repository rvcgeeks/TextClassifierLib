#include "NaiveBayesClassifier.h"

NaiveBayesClassifier::NaiveBayesClassifier()
{
    CV.setBinary(false);
    CV.setCaseSensitive(false);
    CV.setIncludeStopWords(false);
}

NaiveBayesClassifier::~NaiveBayesClassifier()
{
}

void NaiveBayesClassifier::shape()
{
    CV.shape();
}
void NaiveBayesClassifier::head()
{
    CV.head();
}

int NaiveBayesClassifier::totalWords()
{
    int ret = 0;
    for (auto sentence : CV.sentences)
    {
        for (auto count_ : sentence->sentence_array)
        {
            ret += count_;
        }
    }
    return ret;
}

int NaiveBayesClassifier::totalWordsOfType(bool label_)
{
    int ret = 0;
    for (auto sentence : CV.sentences)
    {
        for (auto count_ : sentence->sentence_array)
        {
            if (sentence->label == label_)
            {
                ret += count_;
            }
        }
    }
    return ret;
}

float NaiveBayesClassifier::pOfType(bool label_)
{
    int ttl = totalWords();
    int ttlOfType = totalWordsOfType(label_);
    return (float)ttlOfType / (float)ttl;
}

int NaiveBayesClassifier::countOccurances(string word)
{
    int ret = 0;
    unsigned int word_array_size = CV.getWordArraySize();
    for (unsigned int i = 0; i < word_array_size; i++)
    {
        if (CV.word_array[i] == word)
        {
            for (auto sentence : CV.sentences)
            {
                if (sentence->sentence_array.size() > i)
                {
                    ret += sentence->sentence_array[i];
                }
            }
        }
    }
    return ret;
}

int NaiveBayesClassifier::countOccurancesOfType(string word, bool label_)
{
    int ret = 0;
    unsigned int word_array_size = CV.getWordArraySize();
    for (unsigned int i = 0; i < word_array_size; i++)
    {
        if (CV.word_array[i] == word)
        {
            for (auto sentence : CV.sentences)
            {
                if (sentence->sentence_array.size() > i)
                {
                    if (sentence->label == label_)
                    {
                        ret += sentence->sentence_array[i];
                    }
                }
            }
        }
    }
    return ret;
}

void NaiveBayesClassifier::fit(string abs_filepath_to_features, string abs_filepath_to_labels)
{
    CV.fit(abs_filepath_to_features, abs_filepath_to_labels);

    smoothing_param_m = 1.0;
    smoothing_param_p = 0.5;

	cout << "fitting NaiveBayesClassifier..." << endl;

    total_words_of_type_true = totalWordsOfType(true);
    logp_true = log(pOfType(true));

    total_words_of_type_false = totalWordsOfType(false);
    logp_false = log(pOfType(false));

	cout << "total_words_of_type_true = " << total_words_of_type_true << endl
		 << "logp_true = " << logp_true << endl
		 << "total_words_of_type_false = " << total_words_of_type_false << endl
		 << "logp_false = " << logp_false << endl
		 << "smoothing_param_m = " << smoothing_param_m << endl
		 << "smoothing_param_p = " << smoothing_param_p << endl;
}

float NaiveBayesClassifier::getWeight(vector<string> sentence, bool label_)
{
    float ret;
    int total_words_of_type;
    float mp = smoothing_param_m * smoothing_param_p;
    float m = smoothing_param_m;

    if (label_ == true)
    {
        total_words_of_type = total_words_of_type_true;
        ret = logp_true;
    }
    else
    {
        total_words_of_type = total_words_of_type_false;
        ret = logp_false;
    }

    for (auto word : sentence)
    {
        ret += log(((float)countOccurancesOfType(word, label_) + mp) / ((float)total_words_of_type + m));
    }
    return ret;
}

int NaiveBayesClassifier::predict(string sentence)
{
    GlobalData vars;
    vector<string> processed_input;
    processed_input = CV.buildSentenceVector(sentence); // Encapsulate better
    float trueWeight = getWeight(processed_input, true);
    float falseWeight = getWeight(processed_input, false);
    if (trueWeight < falseWeight)
    {
        return vars.NEG;
    }
    else if (trueWeight > falseWeight)
    {
        return vars.POS;
    }
    else
    {
        return vars.NEU;
    }
}

void NaiveBayesClassifier::predict(string abs_filepath_to_features, string abs_filepath_to_labels)
{
    ifstream in;
    ofstream out;
    string feature_input;
    int label_output;

    in.open(abs_filepath_to_features);
    if (!in)
    {
        cout << "ERROR: Cannot open features file.\n";
        return;
    }

    out.open(abs_filepath_to_labels);
    if (!out)
    {
        cout << "ERROR: Cannot open labels file.\n";
        return;
    }

    while (getline(in, feature_input))
    {
        label_output = predict(feature_input);
        out << label_output << endl;
    }

    in.close();
    out.close();
}

void NaiveBayesClassifier::save(const std::string& filename) const
{
    std::ofstream outFile(filename, std::ios::binary);
    if (!outFile.is_open())
    {
        std::cerr << "Failed to open file for writing." << std::endl;
        return;
    }

    CV.save(outFile);

    outFile.write(reinterpret_cast<const char*>(&total_words_of_type_true), sizeof(total_words_of_type_true));
    outFile.write(reinterpret_cast<const char*>(&logp_true), sizeof(logp_true));
    outFile.write(reinterpret_cast<const char*>(&total_words_of_type_false), sizeof(total_words_of_type_false));
    outFile.write(reinterpret_cast<const char*>(&logp_false), sizeof(logp_false));
    outFile.write(reinterpret_cast<const char*>(&smoothing_param_m), sizeof(smoothing_param_m));
    outFile.write(reinterpret_cast<const char*>(&smoothing_param_p), sizeof(smoothing_param_p));

    outFile.close();
}

void NaiveBayesClassifier::load(const std::string& filename)
{
    std::ifstream inFile(filename, std::ios::binary);
    if (!inFile.is_open())
    {
        std::cerr << "Failed to open file for reading." << std::endl;
        return;
    }

    CV.load(inFile);

    inFile.read(reinterpret_cast<char*>(&total_words_of_type_true), sizeof(total_words_of_type_true));
    inFile.read(reinterpret_cast<char*>(&logp_true), sizeof(logp_true));
    inFile.read(reinterpret_cast<char*>(&total_words_of_type_false), sizeof(total_words_of_type_false));
    inFile.read(reinterpret_cast<char*>(&logp_false), sizeof(logp_false));
    inFile.read(reinterpret_cast<char*>(&smoothing_param_m), sizeof(smoothing_param_m));
    inFile.read(reinterpret_cast<char*>(&smoothing_param_p), sizeof(smoothing_param_p));

    inFile.close();
}
