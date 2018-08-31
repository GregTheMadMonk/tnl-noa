#include <iostream>
#include "tnlRootNode.h"
#include "tnlArea2D.h"
#include "tnlCircle2D.h"

int main()
{
    const unsigned x = 4;
    const unsigned y = 4;
    const unsigned size = x * y;
    tnlArea2D* area = new tnlArea2D( 0, 20, 0, 20 );
    tnlCircle2D* circle = new tnlCircle2D( 10, 10, 4 );
    tnlRootNode< size >* root = new tnlRootNode< size >( area, circle, x, y );
    root->setNode();
    root->printStates();
    return 0;
}
