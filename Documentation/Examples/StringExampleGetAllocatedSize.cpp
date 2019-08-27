#include <iostream>
#include <TNL/String.h>

using namespace TNL;
using namespace std;

int main()
{
    String str("my world");
    cout << "Allocated_size = " << str.getAllocatedSize() << endl;
}