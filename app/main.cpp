
#include <iostream>

#include "../source/NaiveBayesClassifier.h"
#include "../source/LogisticRegressionClassifier.h"
#include "../source/LinearSVCClassifier.h"

using namespace std;

int main(int argc, char **argv)
{
	BaseClassifier* pclsfr;
	
	switch (argv[2][0]) 
	{
		case '1':
			pclsfr = new NaiveBayesClassifier();
			break;

		case '2':
			pclsfr = new LogisticRegressionClassifier();
			break;

		case '3':
			pclsfr = new LinearSVCClassifier();
			break;

		default:
			cerr << "Invalid Classifier!" << endl;
			return 1;
	}
	
	// txtclsfr f 2 my_model.bin features.txt labels.txt
	if(argv[1][0] == 'f') {
		cout << "Training\n";
		pclsfr->fit(argv[4], argv[5]);
		pclsfr->shape();
		pclsfr->save(argv[3]);

	// txtclsfr p 2 my_model.bin features.txt labels_pred.txt
	} else if(argv[1][0] == 'p') {
		cout << "Predicting\n";
		pclsfr->load(argv[3]);
		pclsfr->predict(argv[4], argv[5]);

	// txtclsfr 1 2 my_model.bin "This is string to classify" 
	} else if(argv[1][0] == '1') {
		pclsfr->load(argv[3]);
		cout << pclsfr->predict(argv[4]) << endl;
	}

	delete pclsfr;

	cout << "Done\n";
    return 0;
}
