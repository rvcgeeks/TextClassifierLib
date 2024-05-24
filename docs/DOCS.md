# Text Classification with CountVectorizer

### What is a CountVectorizer?
A CountVectorizer is an text-storage data structure from the popular ML library Scikit Learn. In their words, a CountVectorizer:  "Converts a collection of text documents to a matrix of token counts"  [source](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html)

Conceptually, it looks something like this:

![CountVectorizer](static/conceptually.jpeg "Count Vectorizer")

### How is it implemented?

My implementation has a word array as a header, and a vector of pointers to sentences:

![Implementation](static/implementation.jpeg "Implementation")

### How can it be used for text classification?

The CountVectorizer readily pairs with any number of classification algorithms.  As of the time of this writing, two algorithms have been built: a simple weighted average classifier and a Bayesian classifier (inspired from scikit learn's NaiveBayes model). [source](https://scikit-learn.org/stable/modules/naive_bayes.html)

![algo](static/algo.jpeg "algo")

### How is the code structured?

The CountVectorizer is a standalone class that is only concerned with holding the data.  The classifiers inherit from a base classifier to reduce redundancy.

![Structure](static/structure.jpeg "Structure")

### A note on data

The training data must be in a specific format to be used in this library.  Two files, a "features" and "labels" file must be used as an input for the model to work. Kindly look at sample_data folder alongside this document. The format is this:

![data](static/data.jpeg "data")

#### NOTE: This is part of a dataset that was created for the Paper 'From Group to Individual Labels using Deep Features', Kotzias et. al,. KDD 2015
