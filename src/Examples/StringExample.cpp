#include <iostream>

using namespace TNL
       
int main()
{
    // constructors
    String str1;
    String str2( "xxstringxxx", 2, 3 );
    String str3( str2 );                    // copy constructor
    String str4( 28.4 );                    // converts to string

    cout << "str1:" << str1 << endl;
    cout << "str2:" << str2 << endl;
    cout << "str3:" << str3 << endl;
    cout << "str4:" << str4 << endl;

    // functions
    int size = str3.getSize();
    cout << "size of string:" << size << "bytes" << endl;

    int alloc_size = str3.getAllocatedSize();
    cout << "alloc_size:" << alloc_size << endl;

    int memory = str1.setSize( 256 );
    cout << "memory:" << memory << endl;

    String str
    setter = str.setString( "Something new" );
    cout << "setter:" << setter << endl;

    int getter = str4.getString();
    cout << "getter:" << getter << endl;

    String word( computer ) ;
    third_letter = word[2];
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
    d = d1;
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
    i = 'a' + 'b';
    cout << "i:" << i << endl;
    
    String letter1( "u" );
    if ( letter1 == "u" ) cout << "Letters are the same." << endl;
    
    String letter2( "v" );
    if ( letter2 != "w" ) cout << "Letters are different." << endl;
    
    //Cast to bool operators
    String full( "string" );
    if ( full ) cout << "String is not empty." << endl;
    
    String empty;
    if ( !empty ) cout << "String is empty." << endl;
    
    //replace
    String phrase( "Hakuna matata" );
    new_phrase = phrase.replace( "a", "u", 2 );
    cout << "new_phrase:" << new_phrase << endl;
    
    
}
