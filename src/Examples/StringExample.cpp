#include <iostream>
#include <TNL/String.h>
#include <TNL/Containers/List.h>
#include <TNL/File.h>

using namespace TNL;
using namespace std;

int main( int argc, char* argv[] )
{
   String emptyString;
   String string1( "string 1" );
   String string2( "string 2" );
   String string3( string2 );
   String string4 = convertToString( 28.4 );

   cout << "empytString = " << emptyString << endl;
   cout << "string1 = " << string1 << endl;
   cout << "string2 = " << string2 << endl;
   cout << "string3 = " << string3 << endl;
   cout << "string4 = " << string4 << endl;

   cout << "emptyString size = " << emptyString.getSize() << endl;
   cout << "string1 size = " << string1.getSize() << endl;
   cout << "string1 length = " << string1.getLength() << endl;

   const char* c_string = string1.getString();
   cout << "c_string = " << c_string << endl;

   cout << " 3rd letter of string1 =" << string1[ 2 ] << endl;

   cout << " string1 + string2 = " << string1 + string2 << endl;
   cout << " string1 + \" another string\" = " << string1 + " another string" << endl;
   
   string2 += "another string";
   cout << " string2 = " << string2;
   string2 = "string 2";

   if( string3 == string2 )
      cout << "string3 == string2" << endl;
   if( string1 != string2 )
      cout << "string1 != string2" << endl;

   if( ! emptyString )
      cout << "emptyString is empty" << endl;
   if( string1 )
      cout << "string1 is not empty" << endl;

   /*File myFile;
   myFile.open( "string_save.out", File::out );
   myFile << string1;
   myFile.close();

   myFile.open( "string_save.out", File::in );
   myFile >> string3;
   cout << "string 3 after loading = " << string3 << endl;*/
}
