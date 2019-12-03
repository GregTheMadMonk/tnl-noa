#include <iostream>
#include <TNL/Containers/Vector.h>
#include <TNL/Containers/Array.h>

using namespace TNL;
using namespace std;

int main()
{
    Containers::Vector<int> vector1( 5 );
    vector1 = 0;
    cout << "Does vector contain 1?" << vector1.containsValue( 1 ) << endl;
    cout << "Does vector contain only zeros?" << vector1.containsOnlyValue( 0 ) << endl;

    Containers::Vector<int> vector2( 3 );
    vector2 = 1;
    vector2.swap( vector1 );
    vector2.setElement( 2, 4 );

    cout << "First vector:" << vector1.getData() << endl;
    cout << "Second vector:" << vector2.getData() << endl;

    vector2.reset();
    cout << "Second vector after reset:" << vector2.getData() << endl;

    Containers::Vector<int> vect = { 1, 2, -3, 3 };
    cout << "The smallest element is:" << min( vect ) << endl;
    cout << "The absolute biggest element is:" << max( abs( vect ) ) << endl;
    cout << "Sum of all vector elements:" << sum( vect ) << endl;
    vect *= 2.0;
    cout << "Vector multiplied by 2:" << vect << endl;
}

