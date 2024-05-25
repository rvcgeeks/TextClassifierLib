
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

void NaiveBayesClassifier::fit(string abs_filepath_to_features, string abs_filepath_to_labels)
{
    CV.fit(abs_filepath_to_features, abs_filepath_to_labels);

    smoothing_param_m = 1.0;
    smoothing_param_p = 0.5;

	cout << "fitting NaiveBayesClassifier..." << endl;

	int total_words = CV.totalWords();

    total_words_of_type_true = CV.totalWordsOfType(true);
    logp_true = log((float)total_words_of_type_true / (float)total_words);

    total_words_of_type_false = CV.totalWordsOfType(false);
    logp_false = log((float)total_words_of_type_false / (float)total_words);

	cout << "total_words_of_type_true = " << total_words_of_type_true << endl
		 << "logp_true = " << logp_true << endl
		 << "total_words_of_type_false = " << total_words_of_type_false << endl
		 << "logp_false = " << logp_false << endl
		 << "smoothing_param_m = " << smoothing_param_m << endl
		 << "smoothing_param_p = " << smoothing_param_p << endl;
}

int NaiveBayesClassifier::predict(string sentence)
{
    GlobalData vars;
    vector<string> processed_input;

    processed_input = CV.buildSentenceVector(sentence);

    float trueWeight, falseWeight;
    float mp = smoothing_param_m * smoothing_param_p;
    float m = smoothing_param_m;

	trueWeight = logp_true;
	for (auto word : processed_input)
    {
        trueWeight += log(((float)CV.countOccurancesOfType(word, true) + mp) / ((float)total_words_of_type_true + m));
    }

	falseWeight = logp_false;
	for (auto word : processed_input)
    {
        falseWeight += log(((float)CV.countOccurancesOfType(word, false) + mp) / ((float)total_words_of_type_false + m));
    }
	
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
