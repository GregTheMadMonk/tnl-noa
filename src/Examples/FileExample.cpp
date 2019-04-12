#include <iostream>
#include <TNL/File.h>
#include <TNL/String.h>

using namespace TNL;
using namespace std;

int main()
{
    File file;

    file.open( String("new-file.tnl"), std::ios_base::out );
    String title("'string to file'");
    file << title;
    file.close();

    file.open( String("new-file.tnl"), std::ios_base::in );
    String restoredString;
    file >> restoredString;
    file.close();

    cout << "restored string = " << restoredString <<endl;
}
