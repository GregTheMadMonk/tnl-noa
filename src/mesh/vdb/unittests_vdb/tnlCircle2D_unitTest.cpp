#include <iostream>
#include "tnlCircle2D.h"

using namespace std;

int main()
{   // dost spatnej unittest -- vylepsit
    tnlCircle2D* circle = new tnlCircle2D( 5, 5, 5 );
    cout << "circle->isIntercept(0, 5, 0, 5) == " << 
            circle->isIntercept( 0, 5, 0, 5 ) << endl;
    return 0;
}
