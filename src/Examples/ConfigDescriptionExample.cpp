#include <iostream>
#include <ConfigDescription.h>

using namespace TNL;
using namespace std;
       
int main()
{
    ConfigDescription confd;
    confd.template addEntry< String >("--new-entry","Specific description.");
    confd.template addEntryEnum< String >("option1");
    confd.template addEntryEnum< String >("option2");
    confd.addDelimiter("-----------------------------");
}
