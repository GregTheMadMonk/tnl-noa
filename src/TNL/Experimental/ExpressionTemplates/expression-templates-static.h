#pragma once

#include <iostream>
#include <iomanip>
#include <TNL/Timer.h>
#include "OverloadedOperators.h"  
#include <TNL/Containers/StaticVector.h>

using namespace std;
using namespace TNL;
using namespace TNL::Containers;

int main()
{
    StaticVector< 10, double > sv1( 1.5 );
    StaticVector< 100, double > sv2( 1.5 );
    StaticVector< 500, double > sv3( 1.5 );
    StaticVector< 1000, double > sv4( 1.5 );
    StaticVector< 2000, double > sv5( 1.5 );
    StaticVector< 5000, double > sv6( 1.5 );
    StaticVector< 10, double > svr1( 0.0 );
    StaticVector< 10, double > svr1_( 0.0 );
    StaticVector< 100, double > svr2( 0.0 );
    StaticVector< 100, double > svr2_( 0.0 );
    StaticVector< 500, double > svr3( 0.0 );
    StaticVector< 500, double > svr3_( 0.0 );
    StaticVector< 1000, double > svr4( 0.0 );
    StaticVector< 1000, double > svr4_( 0.0 );
    StaticVector< 2000, double > svr5( 0.0 );
    StaticVector< 2000, double > svr5_( 0.0 );
    StaticVector< 5000, double > svr6( 0.0 );
    StaticVector< 5000, double > svr6_( 0.0 );

    std::vector<double> v1( 10, 1.5 );
    std::vector<double> v2( 100, 1.5 );
    std::vector<double> v3( 500, 1.5 );
    std::vector<double> v4( 1000, 1.5 );
    std::vector<double> v5( 2000, 1.5 );
    std::vector<double> v6( 5000, 1.5 );
    std::vector<double> vr1(10), vr2(100), vr3(500), vr4(1000), vr5(2000), vr6(5000), vr1_(10), vr2_(100), vr3_(500), vr4_(1000), vr5_(2000), vr6_(5000);
    std::vector<double> cvr1(10), cvr2(100), cvr3(500), cvr4(1000), cvr5(2000), cvr6(5000), cvr1_(10), cvr2_(100), cvr3_(500), cvr4_(1000), cvr5_(2000), cvr6_(5000);
    
    TNL::Timer t1;
    TNL::Timer t2;
    TNL::Timer t3;
    
    long double stm1 = 0, stm2 = 0, stm3 = 0, stm4 = 0, stm5 = 0, stm6 = 0, stm1_ = 0, stm2_ = 0, stm3_ = 0, stm4_ = 0, stm5_ = 0, stm6_ = 0;
    long double tm1 = 0, tm2 = 0, tm3 = 0, tm4 = 0, tm5 = 0, tm6 = 0, tm1_ = 0, tm2_ = 0, tm3_ = 0, tm4_ = 0, tm5_ = 0, tm6_ = 0;
    long double ctm1 = 0, ctm2 = 0, ctm3 = 0, ctm4 = 0, ctm5 = 0, ctm6 = 0, ctm1_ = 0, ctm2_ = 0, ctm3_ = 0, ctm4_ = 0, ctm5_ = 0, ctm6_ = 0;
    
    int numb = 50000;
    
    //static vectors
    
    t1.start();
    for( int i = 0; i < numb; i++ )
        svr1 = sv1 + sv1;
    t1.stop();
    stm1 = t1.getCPUCycles();
    
    t1.reset();
    t1.start();
    for( int i = 0; i < numb; i++ )
        svr2 = sv2 + sv2;
    t1.stop();
    stm2 = t1.getCPUCycles();
    
    t1.reset();
    t1.start();
    for( int i = 0; i < numb; i++ )
        svr3 = sv3 + sv3;
    t1.stop();
    stm3 = t1.getCPUCycles();
    
    t1.reset();
    t1.start();
    for( int i = 0; i < numb; i++ )
        svr4 = sv4 + sv4;
    t1.stop();
    stm4 = t1.getCPUCycles();
    
    t1.reset();
    t1.start();
    for( int i = 0; i < numb; i++ )
        svr5 = sv5 + sv5;
    t1.stop();
    stm5 = t1.getCPUCycles();
    
    t1.reset();
    t1.start();
    for( int i = 0; i < numb; i++ )
        svr6 = sv6 + sv6;
    t1.stop();
    stm6 = t1.getCPUCycles();
    
    t1.reset();
    t1.start();
    for( int i = 0; i < numb; i++ )
        svr1_ = sv1 + sv1 + sv1 + sv1 + sv1 + sv1 + sv1 + sv1 + sv1 + sv1;
    t1.stop();
    stm1_ = t1.getCPUCycles();
    
    t1.reset();
    t1.start();
    for( int i = 0; i < numb; i++ )
        svr2_ = sv2 + sv2 + sv2 + sv2 + sv2 + sv2 + sv2 + sv2 + sv2 + sv2;
    t1.stop();
    stm2_ = t1.getCPUCycles();
    
    t1.reset();
    t1.start();
    for( int i = 0; i < numb; i++ )
        svr3_ = sv3 + sv3 + sv3 + sv3 + sv3 + sv3 + sv3 + sv3 + sv3 + sv3;
    t1.stop();
    stm3_ = t1.getCPUCycles();
    
    t1.reset();
    t1.start();
    for( int i = 0; i < numb; i++ )
        svr4_ = sv4 + sv4 + sv4 + sv4 + sv4 + sv4 + sv4 + sv4 + sv4 + sv4;
    t1.stop();
    stm4_ = t1.getCPUCycles();
    
    t1.reset();
    t1.start();
    for( int i = 0; i < numb; i++ )
        svr5_ = sv5 + sv5 + sv5 + sv5 + sv5 + sv5 + sv5 + sv5 + sv5 + sv5;
    t1.stop();
    stm5_ = t1.getCPUCycles();
    
    t1.reset();
    t1.start();
    for( int i = 0; i < numb; i++ )
        svr6_ = sv6 + sv6 + sv6 + sv6 + sv6 + sv6 + sv6 + sv6 + sv6 + sv6;
    t1.stop();
    stm6_ = t1.getCPUCycles();
    
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
    std::cout << "static vectors" << "\t" << stm1 << "\t\t" << stm2 << "\t\t" << stm3 << "\t\t" << stm4 << "\t\t" << stm5 << "\t\t" << stm6 << std::endl;
    std::cout << "overloaded ops." << "\t" << tm1 << "\t" << tm2 << "\t" << tm3 << "\t" << tm4 << "\t" << tm5 << "\t" << tm6 << std::endl;
    std::cout << "pure c" << "\t\t"        << ctm1 << "\t" << ctm2 << "\t" << ctm3 << "\t" << ctm4 << "\t" << ctm5 << "\t" << ctm6  << "\n" << std::endl;

    std::cout << "addition of 10 vectors" << std::endl;
    std::cout << "size\t\t"                  << "10" << "\t\t" << "100" << "\t\t" << "500" << "\t\t" << "1000" << "\t\t" << "2000" << "\t\t" << "5000" << std::endl;
    std::cout << "static vectors" << "\t" << stm1_ << "\t\t" << stm2_ << "\t\t" << stm3_ << "\t\t" << stm4_ << "\t\t" << stm5_ << "\t\t" << stm6_ << std::endl;
    std::cout << "overloaded ops." << "\t" << tm1_ << "\t" << tm2_ << "\t" << tm3_ << "\t" << tm4_ << "\t" << tm5_ << "\t" << tm6_ << std::endl;
    std::cout << "pure c" << "\t\t"        << ctm1_ << "\t" << ctm2_ << "\t" << ctm3_ << "\t" << ctm4_ << "\t" << ctm5_ << "\t" << ctm6_ << std::endl;
    
    return 0;
}
