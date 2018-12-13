#include <iostream>
#include <TNL/String.h>

using namespace TNL;
using namespace std;
       
int main()
{
    String str("my world");
    int size = str.getSize();
    cout << "size of string:" << size << "bytes" << endl;
}

