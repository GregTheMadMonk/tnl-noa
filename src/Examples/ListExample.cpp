#include <iostream>
#include <TNL/Config/ConfigDescription.h>
#include <TNL/Containers/List.h>
#include <TNL/Containers/Array.h>

using namespace TNL;
using namespace std;
       
int main()
{
    Containers::List< int > lst;
    lst.isEmpty();

    lst.Append(1);
    lst.Append(3);

    lst.isEmpty();
    lst.getSize();

    lst.Insert(2,1);

    Containers::Array<int> array;
    lst.toArray(array);
}