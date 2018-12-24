#include <iostream>
#include <TNL/String.h>
#include <TNL/Containers/List.h>
#include <TNL/File.h>

using namespace TNL;
using namespace std;

int main( int argc, char* argv[] )
{
    // constructors
    String str1;
    String str2( "string" );
    String str3( str2 );                    // copy constructor
    String str4 = convertToString( 28.4 );  // converts to string

    cout << "str1:" << str1 << endl;
    cout << "str2:" << str2 << endl;
    cout << "str3:" << str3 << endl;
    cout << "str4:" << str4 << endl;

    // functions
    /*int size = str3.getSize();
    cout << "size of string:" << size << "bytes" << endl;

    int alloc_size = str3.getAllocatedSize();
    cout << "alloc_size:" << alloc_size << endl;

    str1.setSize( 256 );
    size = str3.getSize();
    cout << "size of string:" << size << "bytes" << endl;*/

    String setter = "Something new";
    cout << "setter:" << setter << endl;

    const char* getter = str4.getString();
    cout << "getter:" << getter << endl;

    String word( "computer" ) ;
    char third_letter = word[2];
    cout << "third_letter:" << third_letter << endl;

    // Operators for C Strings
    String a( "hello" );
    a = "bye";
    cout << "a:" << a << endl;

    String b( "see" );
    b += " you";
    cout << "b:" << b << endl;

    String c;
    c = b + " soon";
    cout << "c:" << c << endl;

    String name( "Jack" );
    if ( name == "Jack" ) cout << "Names are the same." << endl;

    String surname( "Sparrow" );
    if ( surname != "Jones" ) cout << "Surnames are different." << endl;

    // Operators for Strings
    String d1( "Cheese" );
    String d = d1;
    cout << "d:" << d << endl;

    String e( "Mac&" );
    e += d;
    cout << "e:" << e << endl;

    String f;
    String f1("Tim likes ");
    f = f1 + e;
    cout << "f:" << f << endl;

    String num1( "one" );
    String num2( "Anyone", 3);
    if ( num1 == num2 ) cout << "Numbers are the same." << endl;

    String eq1( "a + b" );
    String eq2( "a" );
    if ( eq1 != eq2 ) cout << "Equations are different." << endl;

    // Operators for single characters
    String g;
    g = 'y';
    cout << "g:" << g << endl;

    String h( "x" );
    h += g;
    cout << "h:" << h << endl;

    String i;
    i = convertToString( 'a' ) + 'b';
    cout << "i:" << i << endl;

    String letter1( "u" );
    if ( letter1 == "u" ) cout << "Letters are the same." << endl;

    String letter2( "v" );
    if ( letter2 != "w" ) cout << "Letters are different." << endl;

    // Cast to bool operators
    String full( "string" );
    if ( full ) cout << "String is not empty." << endl;

    String empty;
    if ( !empty ) cout << "String is empty." << endl;

    // replace
    String phrase( "Hakuna matata" );
    String new_phrase = phrase.replace( "a", "u", 2 );
    cout << "new_phrase:" << new_phrase << endl;

    // strip
    String names("       Josh Martin   John  Marley Charles   ");
    String better_names = names.strip();
    cout << "better_names:" << better_names << endl;

    // split
    String dates("3/4/2005;8/7/2011;11/12/2019");
    std::vector<String> list = dates.split(';');
    cout << "list_dates: " << list[0] << ", " << list[1] << ", " << list[2] << endl;

    String cars("opel,mazda,,skoda,");
    std::vector<String> list3 = cars.split(',', true);
    cout << "split with true:" << list3[0] << ", " << list3[1] << ", " << list3[2] << endl;
    std::vector<String> list5 = cars.split(',');
    cout << "split with false:" << list5[0] << ", " << list5[1] << ", " << list5[2] << ", " << list5[3] << endl;

    // save
    File myFile;
    myFile << String("Header"); // saves "Header" into myFile

    // load
    String strg;
    myFile >> strg;
    cout << "strg:" << strg << endl;
}
