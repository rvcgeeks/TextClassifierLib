#include <iostream>
#include "../source/NaiveBayesClassifier.h"

using namespace std;

int main(int argc, char **argv)
{
	NaiveBayesClassifier clsfr;
	
	// txtclsfr f my_model.bin features.txt labels.txt
	if(argv[1][0] == 'f') {
		cout << "Training\n";
		clsfr.fit(argv[3], argv[4]);
		clsfr.shape();
		clsfr.save(argv[2]);

	// txtclsfr p my_model.bin features.txt labels_pred.txt
	} else if(argv[1][0] == 'p') {
		cout << "Predicting\n";
		clsfr.load(argv[2]);
		clsfr.predict(argv[3], argv[4]);

	// txtclsfr 1 my_model.bin "This is string to classify" 
	} else if(argv[1][0] == '1') {
		clsfr.load(argv[2]);
		cout << clsfr.predict(argv[3]) << endl;
	}

	cout << "Done\n";
    return 0;
}