/**
 * @file BaseClassifier.cpp
 * @brief Implementation of the BaseClassifier class.
 */

#include "BaseClassifier.h"

/**
 * @brief Constructor for BaseClassifier.
 */
BaseClassifier::BaseClassifier()
{
}

/**
 * @brief Destructor for BaseClassifier.
 */
BaseClassifier::~BaseClassifier()
{
}

/**
 * @brief Display the shape of the dataset.
 */
void BaseClassifier::shape()
{
    pVec->shape();
}

/**
 * @brief Display the head of the dataset.
 */
void BaseClassifier::head()
{
    pVec->head();
}
