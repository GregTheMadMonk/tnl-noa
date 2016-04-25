#include <iostream>
#include <fstream>
#include "tnlRootNode.h"
#include "tnlArea2D.h"
#include "tnlCircle2D.h"

int main()
{
    const unsigned x = 4;
    const unsigned y = 4;
    const unsigned size = x * y;
    tnlArea2D< double >* area = new tnlArea2D< double >( 0, 20, 0, 20 );
    tnlCircle2D< double >* circle = new tnlCircle2D< double >( 10, 10, 4 );
    tnlRootNode< double, int, size, x, y >* root = new tnlRootNode< double, int, size, x, y >( area, circle, x, y, 5 );
    root->createTree();
    //root->printStates( f );
    root->write();
    return 0;
}
