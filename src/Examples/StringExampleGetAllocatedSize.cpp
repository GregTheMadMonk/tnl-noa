#include <iostream>
#include <TNL/String.h>

using namespace TNL;
using namespace std;
       
int main()
{
    String str("my world");
    int alloc_size = str.getAllocatedSize();
    cout << "alloc_size:" << alloc_size << endl;
}