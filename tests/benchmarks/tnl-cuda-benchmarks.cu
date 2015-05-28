/***************************************************************************
                          tnl-cuda-benchmarks.cu  -  description
                             -------------------
    begin                : May 28, 2015
    copyright            : (C) 2015 by Tomas Oberhuber
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

#include <core/vectors/tnlVector.h>
#include <core/tnlTimerRT.h>

int main( int argc, char* argv[] )
{
#ifdef HAVE_CUDA

    tnlTimerRT timer;
    const double oneGB = 1024.0 * 1024.0 * 1024.0;

    cout << "Benchmarking memory bandwidth when transfering int ..." << endl;

    const int size = 1 << 22;
    
    tnlVector< int, tnlHost > hostVector;
    tnlVector< int, tnlCuda > deviceVector;
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

   
#endif
   return EXIT_SUCCESS;
}
