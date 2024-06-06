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

class CountVectorizer : public BaseVectorizer
{
public:
	// ======================CONSTRUCTORS==============================|

	// Default constructor has two options, one which takes no params,
	// which defaults to case_sensitive=true and include_stopwords=true:
	CountVectorizer();

	// And another which allows the user to choose their options:
	CountVectorizer(bool binary_, bool case_sensitive_, bool include_stopwords_);

	// Destructor:
	~CountVectorizer();

	// ======================USER INTERFACE FUNCTIONS==================|

	// Fit will add additional (labeled) data to a CV object.  User must
	// provide an absolute filepath to the features and an absolute file-
	// path to the labels:
	void fit(string abs_filepath_to_features, string abs_filepath_to_labels) override;

	// Prints the dimensions of the CV object:
	void shape() override;

	// Prints a dictionary-like representation of the CV object (first 10):
	void head() override;

	// ======================HELPERS===================================|

	// Checks if a word is in the sentence.  Returns an integer casted bool:
	int is_wordInSentence(Sentence sentence_, unsigned int idx);

	// Given a sentence, pushSentenceToWordArray will update the CV's word_array
	// to incorporate newly discovered words:
	void pushSentenceToWordArray(vector<string> new_sentence_vector);

	// Given a sentence, return a fully constructed sentence object:
	shared_ptr<Sentence> createSentenceObject(vector<string> new_sentence_vector, bool label_);

	// Given a sentence, add the sentence to the CountVectorizer.  Combines
	// buildSentenceVector, pushSentenceToWordArray, and createSentenceObject
	// to accomplish this task:
	void addSentence(string new_sentence, bool label_) override;

	// Checks if the CV object already contains the word.
	bool ContainsWord(const string& word_to_check) override;

	// Given a sentence, split the sentence into a vector of words.
	// Punctuation should be its own element.
	vector<string> buildSentenceVector(string sentence_) override;

	// Private attribute setter functions:
	std::vector<double> getSentenceFeatures(std::vector<std::string> sentence_words) const override;

	// Functions to load and save model
	void save(std::ofstream& outFile) const override;
	void load(std::ifstream& inFile) override;
};

#endif // COUNTERVECTORIZER_H__
