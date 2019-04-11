#include <iostream>
#include <TNL/Object.h>
#include <unistd.h>

using namespace TNL;
using namespace std;

int main()
{
   auto parsedObjectType = parseObjectType( String( "MyObject< Value, Device, Index >" ) );
   for( auto &token : parsedObjectType )
   {
      cout << token << endl;
   }
}

