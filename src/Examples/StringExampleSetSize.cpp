#include <iostream>
#include <TNL/String.h>

using namespace TNL;
using namespace std;
       
int main()
{
    String memory;
    memory.setSize( 256 );
    int memorysize = memory.getSize();
    cout << "memory:" << memorysize << endl;
}
