#include <iostream>
#include <TNL/Containers/Array.h>

using namespace TNL;
using namespace std;

int main()
{
    Containers::Array<int> array1;
    array1.setSize(5);
    array1.setValue(0);
    cout << "Does array contain 1?" << array1.containsValue(1) << endl;
    cout << "Does array contain only zeros?" << array1.containsOnlyValue(0) << endl;

    Containers::Array<int> array2(3);
    array2.setValue(1);
    array2.swap(array1);
    array2.setElement(2,4);

    cout << "First array:" << array1.getData() << endl;
    cout << "Second array:" << array2.getData() << endl;

    array2.reset();
    cout << "Second array after reset:" << array2.getData() << endl;

    // bind
}
