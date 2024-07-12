/**
 * @file TextClassifierFactory.h
 * @brief Header file containing the declaration of TextClassifierFactory class.
 */

#ifndef TEXTCLASSIFIERFACTORY_H__
#define TEXTCLASSIFIERFACTORY_H__

#include <memory>

using namespace std;

#include "NaiveBayesClassifier.h"
#include "LogisticRegressionClassifier.h"
#include "SVCClassifier.h"
#include "KNNClassifier.h"
#include "RandomForestClassifier.h"
#include "GradientBoostingClassifier.h"

/**
 * @brief A factory class for creating text classifiers.
 */
class TextClassifierFactory
{
public:
    /**
     * @brief Constructor for TextClassifierFactory.
     */
    TextClassifierFactory();

    /**
     * @brief Destructor for TextClassifierFactory.
     */
    ~TextClassifierFactory();

    /**
     * @brief Typedef for the product of the factory, which is a shared pointer to BaseClassifier.
     */
	typedef std::tr1::shared_ptr<BaseClassifier> Product;

    /**
     * @brief Creates a text classifier based on the given vectorizer and classifier IDs.
     * 
     * @param vectorizer_id The ID of the vectorizer to be used.
     * @param classifier_id The ID of the classifier to be used.
     * @return A shared pointer to the created text classifier.
     */
    Product getTextClassifier(int vectorizer_id, int classifier_id);
};

template<typename T, typename Arg1>
std::tr1::shared_ptr<T> make_shared(Arg1 arg1) {
    return std::tr1::shared_ptr<T>(new T(arg1));
}

#endif // TEXTCLASSIFIERFACTORY_H__
