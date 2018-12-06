#include <iostream>
#include <TNL/ConfigDescription.h>

using namespace TNL;
using namespace std;
       
int main()
{
    template List< int > lst;
    List lst;
    lst.isEmpty();

    lst.Append(1);
    lst.Append(3);

    lst.isEmpty();
    lst.getSize();

    lst.Insert(2,1);

    Array array;
    lst.template toArray< int >(array);
}