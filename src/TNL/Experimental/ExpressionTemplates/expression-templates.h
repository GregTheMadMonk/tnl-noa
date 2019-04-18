#pragma once

#include <iostream>
#include <iomanip>
#include <TNL/Timer.h>
#include "OverloadedOperators.h"
#include <TNL/Containers/Vector.h>
#include <TNL/Experimental/ExpressionTemplates/VectorExpressions.h>
//#include <TNL/Experimental/ExpressionTemplates/VectorExpressionsWithReferences.h>

using namespace std;
using namespace TNL;
using namespace TNL::Containers;

int main()
{   
    
    Vector< double, Devices::Host, int > d1( 10 );
    for( int i = 0; i < 10; i++)
        d1[i] = 1.5;
    Vector< double, Devices::Host, int > d2( 100 );
    for( int i = 0; i < 100; i++)
        d2[i] = 1.5;
    Vector< double, Devices::Host, int > d3( 500 );
    for( int i = 0; i < 500; i++)
        d3[i] = 1.5;
    Vector< double, Devices::Host, int > d4( 1000 );
    for( int i = 0; i < 1000; i++)
        d4[i] = 1.5;
    Vector< double, Devices::Host, int > d5( 2000 );
        for( int i = 0; i < 2000; i++)
        d5[i] = 1.5;
    Vector< double, Devices::Host, int > d6( 5000 );
        for( int i = 0; i < 5000; i++)
        d6[i] = 1.5;
    Vector< double, Devices::Host, int > dr1( 10 );
    Vector< double, Devices::Host, int > dr2( 100 );
    Vector< double, Devices::Host, int > dr3( 500 );
    Vector< double, Devices::Host, int > dr4( 1000 );
    Vector< double, Devices::Host, int > dr5( 2000 );
    Vector< double, Devices::Host, int > dr6( 5000 );
    
    VectorView< double, Devices::Host, int > dv1( d1 );
    VectorView< double, Devices::Host, int > dv2( d2 );
    VectorView< double, Devices::Host, int > dv3( d3 );
    VectorView< double, Devices::Host, int > dv4( d4 );
    VectorView< double, Devices::Host, int > dv5( d5 );
    VectorView< double, Devices::Host, int > dv6( d6 );
    VectorView< double, Devices::Host, int > dvr1( dr1 );
    VectorView< double, Devices::Host, int > dvr2( dr2 );
    VectorView< double, Devices::Host, int > dvr3( dr3 );
    VectorView< double, Devices::Host, int > dvr4( dr4 );
    VectorView< double, Devices::Host, int > dvr5( dr5 );
    VectorView< double, Devices::Host, int > dvr6( dr6 );
    VectorView< double, Devices::Host, int > dvr1_( dr1 );
    VectorView< double, Devices::Host, int > dvr2_( dr2 );
    VectorView< double, Devices::Host, int > dvr3_( dr3 );
    VectorView< double, Devices::Host, int > dvr4_( dr4 );
    VectorView< double, Devices::Host, int > dvr5_( dr5 );
    VectorView< double, Devices::Host, int > dvr6_( dr6 );
    
    std::vector<double> v1( 10, 1.5 );
    std::vector<double> v2( 100, 1.5 );
    std::vector<double> v3( 500, 1.5 );
    std::vector<double> v4( 1000, 1.5 );
    std::vector<double> v5( 2000, 1.5 );
    std::vector<double> v6( 5000, 1.5 );
    std::vector<double> vr1(10), vr2(100), vr3(500), vr4(1000), vr5(2000), vr6(5000), vr1_(10), vr2_(100), vr3_(500), vr4_(1000), vr5_(2000), vr6_(5000);
    std::vector<double> cvr1(10), cvr2(100), cvr3(500), cvr4(1000), cvr5(2000), cvr6(5000), cvr1_(10), cvr2_(100), cvr3_(500), cvr4_(1000), cvr5_(2000), cvr6_(5000);
    
    TNL::Timer t2;
    TNL::Timer t3;
    TNL::Timer t4;
    
    long double dtm1 = 0, dtm2 = 0, dtm3 = 0, dtm4 = 0, dtm5 = 0, dtm6 = 0, dtm1_ = 0, dtm2_ = 0, dtm3_ = 0, dtm4_ = 0, dtm5_ = 0, dtm6_ = 0;
    long double tm1 = 0, tm2 = 0, tm3 = 0, tm4 = 0, tm5 = 0, tm6 = 0, tm1_ = 0, tm2_ = 0, tm3_ = 0, tm4_ = 0, tm5_ = 0, tm6_ = 0;
    long double ctm1 = 0, ctm2 = 0, ctm3 = 0, ctm4 = 0, ctm5 = 0, ctm6 = 0, ctm1_ = 0, ctm2_ = 0, ctm3_ = 0, ctm4_ = 0, ctm5_ = 0, ctm6_ = 0;
    
    int numb = 50000;
    
    //dynamic vectors
    
    t4.start();
    for( int i = 0; i < numb; i++ )
        dvr1.evaluate( dv1 + dv1 );
    t4.stop();
    dtm1 = t4.getCPUCycles();
    
    t4.reset();
    t4.start();
    for( int i = 0; i < numb; i++ )
        dvr2.evaluate( dv2 + dv2 );
    t4.stop();
    dtm2 = t4.getCPUCycles();
    
    t4.reset();
    t4.start();
    for( int i = 0; i < numb; i++ )
        dvr3.evaluate( dv3 + dv3 );
    t4.stop();
    dtm3 = t4.getCPUCycles();
    
    t4.reset();
    t4.start();
    for( int i = 0; i < numb; i++ )
        dvr4.evaluate( dv4 + dv4 );
    t4.stop();
    dtm4 = t4.getCPUCycles();
  
    t4.reset();
    t4.start();
    for( int i = 0; i < numb; i++ )
        dvr5.evaluate( dv5 + dv5 );
    t4.stop();
    dtm5 = t4.getCPUCycles();
        
    t4.reset();
    t4.start();
    for( int i = 0; i < numb; i++ )
        dvr6.evaluate( dv6 + dv6 );
    t4.stop();
    dtm6 = t4.getCPUCycles();
        
    t4.reset();
        t4.start();
    for( int i = 0; i < numb; i++ )
        dvr1_.evaluate( dv1 + dv1 + dv1 + dv1 + dv1 + dv1 + dv1 + dv1 + dv1 + dv1 );
    t4.stop();
    dtm1_ = t4.getCPUCycles();
    
    t4.reset();
    t4.start();
    for( int i = 0; i < numb; i++ )
        dvr2_.evaluate( dv2 + dv2 + dv2 + dv2 + dv2 + dv2 + dv2 + dv2 + dv2 + dv2 );
    t4.stop();
    dtm2_ = t4.getCPUCycles();
    
    t4.reset();
    t4.start();
    for( int i = 0; i < numb; i++ )
        dvr3_.evaluate( dv3 + dv3 + dv3 + dv3 + dv3 + dv3 + dv3 + dv3 + dv3 + dv3 );
    t4.stop();
    dtm3_ = t4.getCPUCycles();
    
    t4.reset();
    t4.start();
    for( int i = 0; i < numb; i++ )
        dvr4_.evaluate( dv4 + dv4 + dv4 + dv4 + dv4 + dv4 + dv4 + dv4 + dv4 + dv4 );
    t4.stop();
    dtm4_ = t4.getCPUCycles();
  
    t4.reset();
    t4.start();
    for( int i = 0; i < numb; i++ )
        dvr5_.evaluate( dv5 + dv5 + dv5 + dv5 + dv5 + dv5 + dv5 + dv5 + dv5 + dv5 );
    t4.stop();
    dtm5_ = t4.getCPUCycles();
        
    t4.reset();
    t4.start();
    for( int i = 0; i < numb; i++ )
        dvr6_.evaluate( dv6 + dv6 + dv6 + dv6 + dv6 + dv6 + dv6 + dv6 + dv6 + dv6 );
    t4.stop();
    dtm6_ = t4.getCPUCycles();

    
    //overloaded operators
    
    t2.start();
    for( int i = 0; i < numb; i++ )
        vr1 = v1 + v1;
    t2.stop();
    tm1 = t2.getCPUCycles();

    t2.reset();
    t2.start();
    for( int i = 0; i < numb; i++ )
        vr2 = v2 + v2;
    t2.stop();
    tm2 = t2.getCPUCycles();

    t2.reset();
    t2.start();
    for( int i = 0; i < numb; i++ )
        vr3 = v3 + v3;
    t2.stop();
    tm3 = t2.getCPUCycles();

    t2.reset();
    t2.start();
    for( int i = 0; i < numb; i++ )
        vr4 = v4 + v4;
    t2.stop();
    tm4 = t2.getCPUCycles();
    
    t2.reset();
    t2.start();
    for( int i = 0; i < numb; i++ )
        vr5 = v5 + v5;
    t2.stop();
    tm5 = t2.getCPUCycles();
    
    t2.reset();
    t2.start();
    for( int i = 0; i < numb; i++ )
        vr6 = v6 + v6;
    t2.stop();
    tm6 = t2.getCPUCycles();
        
    t2.reset();
    t2.start();
    for( int i = 0; i < numb; i++ )
        vr1_ = v1 + v1 + v1 + v1 + v1 + v1 + v1 + v1 + v1 + v1;
    t2.stop();
    tm1_ = t2.getCPUCycles();

    t2.reset();
    t2.start();
    for( int i = 0; i < numb; i++ )
        vr2_ = v2 + v2 + v2 + v2 + v2 + v2 + v2 + v2 + v2 + v2;
    t2.stop();
    tm2_ = t2.getCPUCycles();

    t2.reset();
    t2.start();
    for( int i = 0; i < numb; i++ )
        vr3_ = v3 + v3 + v3 + v3 + v3 + v3 + v3 + v3 + v3 + v3;
    t2.stop();
    tm3_ = t2.getCPUCycles();

    t2.reset();
    t2.start();
    for( int i = 0; i < numb; i++ )
        vr4_ = v4 + v4 + v4 + v4 + v4 + v4 + v4 + v4 + v4 + v4;
    t2.stop();
    tm4_ = t2.getCPUCycles();
    
    t2.reset();
    t2.start();
    for( int i = 0; i < numb; i++ )
        vr5_ = v5 + v5 + v5 + v5 + v5 + v5 + v5 + v5 + v5 + v5;
    t2.stop();
    tm5_ = t2.getCPUCycles();
    
    t2.reset();
    t2.start();
    for( int i = 0; i < numb; i++ )
        vr6_ = v6 + v6 + v6 + v6 + v6 + v6 + v6 + v6 + v6 + v6;
    t2.stop();
    tm6_ = t2.getCPUCycles();

    //pure c
    
    t3.start();
    for( int i = 0; i < numb; i++ ){
        for( unsigned int i = 0; i < v1.size(); ++i)
        {
            cvr1[ i ] = v1[ i ] + v1[ i ];
        }
    }
    t3.stop();
    ctm1 = t3.getCPUCycles();

    t3.reset();
    t3.start();
    for( int i = 0; i < numb; i++ ){
        for( unsigned int i = 0; i < v2.size(); ++i)
        {
            cvr2[ i ] = v2[ i ] + v2[ i ];
        }
    }
    t3.stop();
    ctm2 = t3.getCPUCycles();

    t3.reset();
    t3.start();
    for( int i = 0; i < numb; i++ ){
        for( unsigned int i = 0; i < v3.size(); ++i)
        {
            cvr3[ i ] = v3[ i ] + v3[ i ];
        }
    }
    t3.stop();
    ctm3 = t3.getCPUCycles();

    t3.reset();
    t3.start();
    for( int i = 0; i < numb; i++ ){
        for( unsigned int i = 0; i < v4.size(); ++i)
        {
            cvr4[ i ] = v4[ i ] + v4[ i ];
        }
    }
    t3.stop();
    ctm4 = t3.getCPUCycles();
    
    t3.reset();
    t3.start();
    for( int i = 0; i < numb; i++ ){
        for( unsigned int i = 0; i < v5.size(); ++i)
        {
            cvr5[ i ] = v5[ i ] + v5[ i ];
        }
    }
    t3.stop();
    ctm5 = t3.getCPUCycles();
    
    t3.reset();
    t3.start();
    for( int i = 0; i < numb; i++ ){
        for( unsigned int i = 0; i < v6.size(); ++i)
        {
            cvr6[ i ] = v6[ i ] + v6[ i ];
        }
    }
    t3.stop();
    ctm6 = t3.getCPUCycles();
    
    t3.reset();
    t3.start();
    for( int i = 0; i < numb; i++ ){
        for( unsigned int i = 0; i < v1.size(); ++i)
        {
            cvr1_[ i ] = v1[ i ] + v1[ i ] + v1[ i ] + v1[ i ] + v1[ i ] + v1[ i ] + v1[ i ] + v1[ i ] + v1[ i ] + v1[ i ];
        }
    }
    t3.stop();
    ctm1_ = t3.getCPUCycles();

    t3.reset();
    t3.start();
    for( int i = 0; i < numb; i++ ){
        for( unsigned int i = 0; i < v2.size(); ++i)
        {
            cvr2_[ i ] = v2[ i ] + v2[ i ] + v2[ i ] + v2[ i ] + v2[ i ] + v2[ i ] + v2[ i ] + v2[ i ] + v2[ i ] + v2[ i ];
        }
    }
    t3.stop();
    ctm2_ = t3.getCPUCycles();

    t3.reset();
    t3.start();
    for( int i = 0; i < numb; i++ ){
        for( unsigned int i = 0; i < v3.size(); ++i)
        {
            cvr3_[ i ] = v3[ i ] + v3[ i ] + v3[ i ] + v3[ i ] + v3[ i ] + v3[ i ] + v3[ i ] + v3[ i ] + v3[ i ] + v3[ i ];
        }
    }
    t3.stop();
    ctm3_ = t3.getCPUCycles();

    t3.reset();
    t3.start();
    for( int i = 0; i < numb; i++ ){
        for( unsigned int i = 0; i < v4.size(); ++i)
        {
            cvr4_[ i ] = v4[ i ] + v4[ i ] + v4[ i ] + v4[ i ] + v4[ i ] + v4[ i ] + v4[ i ] + v4[ i ] + v4[ i ] + v4[ i ];
        }
    }
    t3.stop();
    ctm4_ = t3.getCPUCycles();
    
    t3.reset();
    t3.start();
    for( int i = 0; i < numb; i++ ){
        for( unsigned int i = 0; i < v5.size(); ++i)
        {
            cvr5_[ i ] = v5[ i ] + v5[ i ] + v5[ i ] + v5[ i ] + v5[ i ] + v5[ i ] + v5[ i ] + v5[ i ] + v5[ i ] + v5[ i ];
        }
    }
    t3.stop();
    ctm5_ = t3.getCPUCycles();
    
    t3.reset();
    t3.start();
    for( int i = 0; i < numb; i++ ){
        for( unsigned int i = 0; i < v6.size(); ++i)
        {
            cvr6_[ i ] = v6[ i ] + v6[ i ] + v6[ i ] + v6[ i ] + v6[ i ] + v6[ i ] + v6[ i ] + v6[ i ] + v6[ i ] + v6[ i ];
        }
    }
    t3.stop();
    ctm6_ = t3.getCPUCycles();
    
    std::cout << std::fixed;
    std::cout << std::setprecision(2);
    
    std::cout << "addition of 2 vectors" << std::endl;
    std::cout << "size\t\t"                  << "10" << "\t\t" << "100" << "\t\t" << "500" << "\t\t" << "1000" << "\t\t" << "2000" << "\t\t" << "5000" << std::endl;
    std::cout << "dynamic vectors" << "\t" << dtm1 << "\t" << dtm2 << "\t" << dtm3 << "\t" << dtm4 << "\t" << dtm5 << "\t" << dtm6 << std::endl;
    std::cout << "overloaded ops." << "\t" << tm1 << "\t" << tm2 << "\t" << tm3 << "\t" << tm4 << "\t" << tm5 << "\t" << tm6 << std::endl;
    std::cout << "pure c" << "\t\t"        << ctm1 << "\t" << ctm2 << "\t" << ctm3 << "\t" << ctm4 << "\t" << ctm5 << "\t" << ctm6  << "\n" << std::endl;

    std::cout << "addition of 10 vectors" << std::endl;
    std::cout << "size\t\t"                  << "10" << "\t\t" << "100" << "\t\t" << "500" << "\t\t" << "1000" << "\t\t" << "2000" << "\t\t" << "5000" << std::endl;
    std::cout << "dynamic vectors" << "\t" << dtm1_ << "\t" << dtm2_ << "\t" << dtm3_ << "\t" << dtm4_ << "\t" << dtm5_ << "\t" << dtm6_ << std::endl;
    std::cout << "overloaded ops." << "\t" << tm1_ << "\t" << tm2_ << "\t" << tm3_ << "\t" << tm4_ << "\t" << tm5_ << "\t" << tm6_ << std::endl;
    std::cout << "pure c" << "\t\t"        << ctm1_ << "\t" << ctm2_ << "\t" << ctm3_ << "\t" << ctm4_ << "\t" << ctm5_ << "\t" << ctm6_ << std::endl;
    
    return 0;
}
