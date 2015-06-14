/***************************************************************************
                          tnl-benchmarks.h  -  description
                             -------------------
    begin                : Jan 27, 2010
    copyright            : (C) 2010 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/

#ifndef TNLCUDABENCHMARKS_H_
#define TNLCUDBENCHMARKS_H_

#include <core/vectors/tnlVector.h>
#include <core/tnlTimerRT.h>

int main( int argc, char* argv[] )
{
#ifdef HAVE_CUDA

    tnlTimerRT timer;
    const double oneGB = 1024.0 * 1024.0 * 1024.0;

    cout << "Benchmarking memory bandwidth: ";

    const int size = 1 << 22;
    
    typedef tnlVector< double, tnlHost > HostVector;
    typedef tnlVector< double, tnlCuda > CudaVector;

    HostVector hostVector;
    CudaVector deviceVector;
    hostVector.setSize( size );
    deviceVector.setSize( size );

    hostVector.setValue( 1.0 );
    deviceVector.setValue( 0.0 );
    
    timer.reset();
    timer.start();
    deviceVector = hostVector;
    timer.stop();    
    double bandwidth = ( double ) ( size ) * sizeof( int ) / timer.getTime() / oneGB;
    cout << bandwidth << " GB/sec." << endl;
    
    HostVector hostVector2;
    CudaVector deviceVector2;
    hostVector2.setLike( hostVector );
    deviceVector2.setLike( deviceVector );
    hostVector2.setValue( 1.0 );
    deviceVector2.setValue( 1.0 );
    cout << "Benchmarking vector addition on CPU: ";
    timer.reset();
    timer.start();
    hostVector.addVector( hostVector2 );
    timer.stop();
    double hostTime = timer.getTime();

#endif
   return EXIT_SUCCESS;
}

#endif /* TNLCUDABENCHMARKS_H_ */
