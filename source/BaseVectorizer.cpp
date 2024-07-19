
/*++

Revision History:
	Date:	Jun 28, 2024.
	Author:	Rajas Chavadekar.
	Desc:	Created.

--*/

#include <iostream>
#include <string>
#include <algorithm>

#include "BaseVectorizer.h"

std::string preprocess_text(const std::string& text) {
    
    std::string processed = text; //.substr(0, MAX_TEXT_LEN);

    std::replace(processed.begin(), processed.end(), '\n', ' ');

    std::string filtered = "";
    for (char c : processed) {
        if (std::isalnum(c) || std::isspace(c) || std::ispunct(c)) {
            filtered += std::tolower(c);
        }
        else
        {
            filtered += ' ';
        }
    }

    return filtered;
}

/**
 * @brief Split a sentence into a vector of words.
 *
 * @param sentence_ The sentence to split.
 * @return Vector of words.
 */
vector<string> BaseVectorizer::buildSentenceVector(string sentence_, bool preprocess)
{
    GlobalData vars;
    string new_word = "";
    vector<string> ret;

    if (true == preprocess)
    {
        sentence_ = preprocess_text(sentence_);
    }

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

void BaseVectorizer::scanForSparseHistogram(std::string abs_filepath_to_features, int minfrequency)
{
    ifstream in;
    string feature;
    vector<string> features;

    in.open(abs_filepath_to_features);

    if (!in)
    {
        cout << "ERROR: Cannot open features file.\n";
        return;
    }

    while (getline(in, feature))
    {
        features = buildSentenceVector(feature);
        for (const auto& x : features)
        {
            if (histogram.count(x) || x.length() == 1)
            {
                histogram[x]++;
            }
            else
            {
                histogram[x] = 1;
            }
        }
    }
    in.close();

    for (const auto& entry : histogram)
    {
        if (entry.second >= minfrequency)
        {
            histogram.erase(entry.first);
        }
    }

    std::cout << "No of Rare Words = " << histogram.size() << std::endl;
}

void BaseVectorizer::setVersionInfo(char* vers_info_in)
{
    memset(vers_info, 0, sizeof(vers_info));
    strcpy(vers_info, vers_info_in);
}
