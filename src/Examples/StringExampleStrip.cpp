#include <iostream>
#include <TNL/String.h>

using namespace TNL;
using namespace std;

int main()
{
    String names(  "       Josh Martin   John  Marley Charles   " );
    String names2( ".......Josh Martin...John..Marley.Charles..." );
    cout << "names strip is: " << names.strip() << endl;
    cout << "names2 strip is: " << names.strip( '.' ) << endl;
}