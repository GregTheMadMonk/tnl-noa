#include <iostream>
#include <fstream>
#include <cstdlib>
#include <ctime>
#include "tnlRootNode.h"
#include "tnlArea2D.h"
#include "tnlCircle2D.h"

int main( int argc,  char** argv )
{
    clock_t begin = clock();
    int areaStart = atoi( argv[ 1 ] );
    int areaEnd = atoi( argv[ 2 ] );
    int circleX = atoi( argv[ 3 ] );
    int circleY = atoi( argv[ 4 ] );
    int radius = atoi( argv[ 5 ] );
    const unsigned x = 4;
    const unsigned y = 4;
    const unsigned size = x * y;
    tnlArea2D< double >* area = new tnlArea2D< double >( areaStart, areaEnd, areaStart, areaEnd );
    tnlCircle2D< double >* circle = new tnlCircle2D< double >( circleX, circleY, radius );
    tnlRootNode< double, int, size, x, y >* root = new tnlRootNode< double, int, size, x, y >( area, circle, x, y, 6 );
    root->createTree();
    clock_t end1 = clock();
    root->write();
    clock_t end2 = clock();
    std::cout << "Tree created in " << ( ( double ) (end1 - begin) ) / CLOCKS_PER_SEC << "s" << std::endl;
    std::cout << "Tree traversed in " << ( ( double )(end2 - begin) ) / CLOCKS_PER_SEC << "s" << std::endl;
    return 0;
}
