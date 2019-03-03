#include <iostream>
#include <TNL/String.h>

using namespace TNL;
using namespace std;

int main()
{
   String phrase( "Say yes yes yes!" );
   cout << "phrase.replace( \"yes\", \"no\", 1 ) = " << phrase.replace( "yes", "no", 1 ) << endl;
   cout << "phrase.replace( \"yes\", \"no\", 2 ) = " << phrase.replace( "yes", "no", 2 ) << endl;
   cout << "phrase.replace( \"yes\", \"no\", 3 ) = " << phrase.replace( "yes", "no", 3 ) << endl;
}