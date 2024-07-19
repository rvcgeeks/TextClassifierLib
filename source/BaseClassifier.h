/**
 * @file BaseClassifier.h
 * @brief Base class for all classifiers.
 */

/*++

Revision History:
	Date:	Jun 28, 2024.
	Author:	Rajas Chavadekar.
	Desc:	Created.

--*/

#ifndef BASECLASSIFIER_H__
#define BASECLASSIFIER_H__

// Comment out if benchmark not needed
#define BENCHMARK

#include <iostream>
#include <sstream>
#include <string>
#ifdef BENCHMARK
#include <ctime>
#endif

#include "CountVectorizer.h"
#include "TfidfVectorizer.h"

using namespace std;
using namespace std::tr1;

#define ID_CLASSIFIER_NAIVEBAYESCLASSIFIER              1
#define ID_CLASSIFIER_LOGISTICREGRESSIONCLASSIFIER      2
#define ID_CLASSIFIER_SVCCLASSIFIER                     3
#define ID_CLASSIFIER_KNNCLASSIFIER                     4
#define ID_CLASSIFIER_RANDOMFORESTCLASSIFIER            5
#define ID_CLASSIFIER_GRADIENTBOOSTINGCLASSIFIER        6

/**
 * @struct Prediction
 * @brief Structure to store prediction results.
 */
struct Prediction
{
    int label;              /**< Predicted label. */
    double probability;     /**< Probability of the predicted label. */
};

/**
 * @class BaseClassifier
 * @brief Abstract base class for classifiers.
 */
class BaseClassifier
{
public:
    /**
     * @brief Constructor for BaseClassifier.
     */
    BaseClassifier();

    /**
     * @brief Destructor for BaseClassifier.
     */
    virtual ~BaseClassifier();

    BaseVectorizer* pVec;   /**< Pointer to the vectorizer. */

    /**
     * @brief Display the shape of the vectorizer.
     */
    void shape();

    /**
     * @brief Display the head of the vectorizer.
     */
    void head();

    /**
     * @brief Set the hyperparameters for the classifier.
     * @param hyperparameters A string representing the hyperparameters.
     */
    virtual void setHyperparameters(std::string hyperparameters) = 0;

    /**
     * @brief Fit the classifier on the given dataset.
     * @param abs_filepath_to_features Absolute file path to the features file.
     * @param abs_filepath_to_labels Absolute file path to the labels file.
     */
    virtual void fit(string abs_filepath_to_features, string abs_filepath_to_labels) = 0;

    /**
     * @brief Predict labels for the given features.
     * @param abs_filepath_to_features Absolute file path to the features file.
     * @param abs_filepath_to_labels Absolute file path to save the predicted labels.
     */
    virtual void predict(string abs_filepath_to_features, string abs_filepath_to_labels, bool preprocess = true) = 0;

    /**
     * @brief Predict the label for a given sentence.
     * @param sentence The input sentence for prediction.
     * @return Prediction containing the label and probability.
     */
    virtual Prediction predict(string sentence, bool preprocess = true) = 0;

    /**
     * @brief Save the classifier to a file.
     * @param filename The name of the file to save the classifier.
     */
    virtual void save(const std::string& filename) const = 0;

    /**
     * @brief Load the classifier from a file.
     * @param filename The name of the file to load the classifier from.
     */
    virtual void load(const std::string& filename) = 0;

	void setVersionInfo(char *vers_info_in);

    int minfrequency;
};

#endif // BASECLASSIFIER_H__
