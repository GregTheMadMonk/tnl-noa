#include <iostream>
#include "tnlCircle2D.h"

using namespace std;

int main()
{   // dost spatnej unittest -- vylepsit
    tnlCircle2D* circle = new tnlCircle2D( 5, 5, 4 );
    cout << "Testing whole circle inside area: ";
    if( circle->isIntercept( 0, 10, 0, 10, true ) )
        cout << "Ok" << endl;
    else
        cout << "Test failed." << endl;

    cout << "Testing whole area inside circle: ";
    if( !circle->isIntercept( 4, 6, 4, 6, true ) )
        cout << "Ok" << endl;
    else
        cout << "Test failed." << endl;

    cout << "Testing left boundry intercept: ";
    if( circle->isIntercept( 3, 7, 0, 2, true ) )
        cout << "Ok" << endl;
    else
        cout << "Test failed." << endl;
    return 0;
}
