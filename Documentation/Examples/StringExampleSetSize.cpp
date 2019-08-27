#include <iostream>
#include <TNL/String.h>

using namespace TNL;
using namespace std;

int main()
{
    String string;
    string.setSize( 1024 );
    cout << "String size = " << string.getSize() << endl;
    cout << "Allocated size = " << string.getAllocatedSize() << endl;
}
