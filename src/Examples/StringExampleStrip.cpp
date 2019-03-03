#include <iostream>
#include <TNL/String.h>

using namespace TNL;
using namespace std;

int main()
{
    String names(  "       Josh Martin   John  Marley Charles   " );
    String names2( ".......Josh Martin...John..Marley.Charles..." );
    cout << "better_names:" << names.strip() << endl;
    cout << "better_names:" << names.strip( '.' ) << endl;
}