#include <iostream>
#include <TNL/Containers/Vector.h>
#include <TNL/Containers/Array.h>

using namespace TNL;
using namespace std;

int main()
{
    Containers::Vector<int> vector1;
    vector1.setSize(5);
    vector1.setValue(0);
    cout << "Does vector contain 1?" << vector1.containsValue(1); << endl;
    cout << "Does vector contain only zeros?" << vector1.containsOnlyValue(0); << endl;

    Containers::Vector<int> vector2(3);
    vector2.setValue(1);
    vector2.swap(vector1);
    vector2.setElement(2,4);

    cout << "First vector:" << vector1.getData() << endl;
    cout << "Second vector:" << vector2.getData() << endl;

    vector2.reset();
    cout << "Second vector after reset:" << vector2.getData() << endl;
}

