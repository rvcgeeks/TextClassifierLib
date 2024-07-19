
/*++

Revision History:
	Date:	Jun 28, 2024.
	Author:	Rajas Chavadekar.
	Desc:	Created.

--*/

#include <iostream>
#include <string>
#include <algorithm>
#include <cctype> // for isalnum, isspace, ispunct, tolower

#include "BaseVectorizer.h"

std::string preprocess_text(const std::string& text) {
    std::string processed = text; // .substr(0, MAX_TEXT_LEN);

    std::replace(processed.begin(), processed.end(), '\n', ' ');

    std::string filtered = "";
    for (std::string::const_iterator it = processed.begin(); it != processed.end(); ++it) {
        char c = *it;
        if (std::isalnum(c) || std::isspace(c) || std::ispunct(c)) {
            filtered += std::tolower(c);
        } else {
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
std::vector<std::string> BaseVectorizer::buildSentenceVector(std::string sentence_, bool preprocess)
{
    GlobalData vars;
    std::string new_word = "";
    std::vector<std::string> ret;

    if (true == preprocess)
    {
        sentence_ = preprocess_text(sentence_);
    }

    for (std::string::iterator it = sentence_.begin(); it != sentence_.end(); ++it)
    {
        char x = *it;
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

    std::vector<std::string> fixed_ret;
    for (std::vector<std::string>::const_iterator it = ret.begin(); it != ret.end(); ++it)
    {
        if (!it->empty())
        {
            fixed_ret.push_back(*it);
        }
    }
    return fixed_ret;
}

void BaseVectorizer::scanForSparseHistogram(std::string abs_filepath_to_features, int minfrequency)
{
    std::ifstream in;
    std::string feature;
    std::vector<std::string> features;

    in.open(abs_filepath_to_features.c_str());

    if (!in)
    {
        std::cout << "ERROR: Cannot open features file.\n";
        return;
    }

    while (std::getline(in, feature))
    {
        features = buildSentenceVector(feature);
        for (std::vector<std::string>::const_iterator it = features.begin(); it != features.end(); ++it)
        {
            const std::string& x = *it;
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

    for (std::tr1::unordered_map<std::string, int>::iterator it = histogram.begin(); it != histogram.end(); )
    {
        if (it->second >= minfrequency)
        {
            it = histogram.erase(it);
        }
        else
        {
            ++it;
        }
    }

    std::cout << "No of Rare Words = " << histogram.size() << std::endl;
}

void BaseVectorizer::setVersionInfo(char *vers_info_in) 
{
	memset(vers_info, 0, sizeof(vers_info));
	strcpy(vers_info, vers_info_in);
}
