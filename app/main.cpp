
#include <iostream>

#include "../source/NaiveBayesClassifier.h"
#include "../source/LogisticRegressionClassifier.h"
#include "../source/SVCClassifier.h"
#include "../source/KNNClassifier.h"
#include "../source/RandomForestClassifier.h"
#include "../source/GradientBoostingClassifier.h"

using namespace std;

int main(int argc, char **argv)
{
	BaseClassifier* pclsfr;

	if (argc < 5)
	{
		cout << "Usage: " << endl
			 << "  " << argv[0] << " f (model id) my_model.bin features.txt labels.txt" << endl
			 << "  " << argv[0] << " p (model id) my_model.bin features.txt labels_pred.txt" << endl
		     << "  " << argv[0] << " 1 (model id) my_model.bin \"This is string to classify\" " << endl
		     << "\nwhere model id = " << endl
			 << "  1 NaiveBayesClassifier\n  2 LogisticRegressionClassifier\n  3 SVCClassifier\n  4 KNNClassifier\n  5 RandomForestClassifier\n  6 GradientBoostingClassifier" << endl;
		return 1;
	}
	
	switch (argv[2][0]) 
	{
		case '1':
			pclsfr = new NaiveBayesClassifier();
			break;

		case '2':
			pclsfr = new LogisticRegressionClassifier();
			break;

		case '3':
			pclsfr = new SVCClassifier();
			break;

		case '4':
			pclsfr = new KNNClassifier();
			break;

		case '5':
			pclsfr = new RandomForestClassifier();
			break;

		case '6':
			pclsfr = new GradientBoostingClassifier();
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
