#include <iostream>

using namespace TNL
       
int main()
{
    String str("my world")
    int alloc_size = str.getAllocatedSize();
    cout << "alloc_size:" << alloc_size << endl;
}