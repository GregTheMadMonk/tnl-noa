#include <iostream>
#include <cstdint>
#include "tnlBitmask.h"

using namespace std;

int main()
{
    for( int i = 0; i < 50000; i++ )
    {
        bool state = i % 2;
        unsigned x = rand() % ( 1 << 30 );
        unsigned y = rand() % ( 1 << 30 );
        tnlBitmask* mask = new tnlBitmask( state, x, y );
        if( state != mask->getState() ||
            x != mask->getX() ||
            y != mask->getY() )
            cout << "state = " << state << ", mask.getState() = " << mask->getState()
            << "x = " << x << ", mask.getX() = " << mask->getX()
            << "y = " << y << ", mask.getY() = " << mask->getY() << endl;
    }
}
