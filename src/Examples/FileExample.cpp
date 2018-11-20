#include <iostream>
#include <TNL/File.h>
#include <TNL/String.h>

using namespace TNL;
using namespace std;
       
int main()
{
    File file;

    file.open( String("new-file.tnl"), IOMode::write );
    String title("Header");
    file.write( title );
    file.close();

    file.open( String("new-file.tnl"), IOMode::read );
    String title2;
    file.read( title2, 4);
    file.close();

    cout << "title2:" << title2 <<endl;
}

