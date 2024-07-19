
/*++

Revision History:
	Date:	Jun 28, 2024.
	Author:	Rajas Chavadekar.
	Desc:	Created.

--*/

#ifndef GLOBALDATA_H__
#define GLOBALDATA_H__

#include <string>
#include <set>
#include <vector>
#include <memory>       // For std::tr1::shared_ptr
#include <unordered_set> // For std::tr1::unordered_set

using namespace std;
using namespace std::tr1;  // Use tr1 namespace for compatibility

class GlobalData
{
public:
   int POS;
   int NEG;
   int NEU;
   int UNK;
   unordered_set<char> punctuation;
   unordered_set<string> stopWords;

   GlobalData()
   {
      POS = 1;
      NEG = 0;
      NEU = -1;
      UNK = -2;

      char punctuationArr[] = "!?/";
      insertElements(&punctuation, punctuationArr, sizeof(punctuationArr) / sizeof(punctuationArr[0]));

      char* stopWordsArr[] = {
          "The", "the", "a", "A", "an", "An",
          "This", "this", "That", "that", "is",
          "Is", "my", "My", ".", ":", ",", ";", "'", ")", "(",
          "..."
      };
      insertElements(&stopWords, stopWordsArr, sizeof(stopWordsArr) / sizeof(stopWordsArr[0]));
   }

private:
   void insertElements(unordered_set<char> *set, char elements[], size_t count)
   {
      for (size_t i = 0; i < count; ++i)
      {
         set->insert(elements[i]);
      }
   }

   void insertElements(unordered_set<string> *set, char** elements, size_t count)
   {
      for (size_t i = 0; i < count; ++i)
      {
         set->insert(string(elements[i]));
      }
   }
};

#endif // GLOBALDATA_H__
