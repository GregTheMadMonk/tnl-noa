#include <iostream>
#include <fstream>
#include "tnlRootNode.h"
#include "tnlArea2D.h"
#include "tnlCircle2D.h"

int main()
{
    fstream f;
    f.open( "mrdat.txt" );
    const unsigned x = 4;
    const unsigned y = 4;
    const unsigned size = x * y;
    tnlArea2D* area = new tnlArea2D( 0, 20, 0, 20 );
    tnlCircle2D* circle = new tnlCircle2D( 10, 10, 4 );
    tnlRootNode< size, x, y >* root = new tnlRootNode< size, x, y >( area, circle, x, y, 5 );
    root->createTree();
    root->printStates( f );
    return 0;
}
