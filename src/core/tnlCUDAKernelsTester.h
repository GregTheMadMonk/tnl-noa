/***************************************************************************
                          tnlCUDAKernelsTester.h
                             -------------------
    begin                : Jan 14, 2010
    copyright            : (C) 2009 by Tomas Oberhuber
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

#ifndef TNLCUDAKERNELSTESTER_H_
#define TNLCUDAKERNELSTESTER_H_

#include <iostream>
#include <cppunit/TestSuite.h>
#include <cppunit/TestResult.h>
#include <cppunit/TestCaller.h>
#include <cppunit/TestCase.h>
#include <core/tnlLongVectorCUDA.h>
#include <core/tnlLongVector.h>
#include <core/mfuncs.h>

using namespace std;

#ifdef HAVE_CUDA
int tnlCUDAReductionMin( const int size,
                         const int block_size,
                         const int grid_size,
                         const int* input );
int tnlCUDAReductionMax( const int size,
                         const int block_size,
                         const int grid_size,
                         const int* input );
int tnlCUDAReductionSum( const int size,
                         const int block_size,
                         const int grid_size,
                         const int* input );
float tnlCUDAReductionMin( const int size,
                           const int block_size,
                           const int grid_size,
                           const float* input );
float tnlCUDAReductionMax( const int size,
                           const int block_size,
                           const int grid_size,
                           const float* input );
float tnlCUDAReductionSum( const int size,
                           const int block_size,
                           const int grid_size,
                           const float* input );
double tnlCUDAReductionMin( const int size,
                            const int block_size,
                            const int grid_size,
                            const double* input );
double tnlCUDAReductionMax( const int size,
                            const int block_size,
                            const int grid_size,
                            const double* input );
double tnlCUDAReductionSum( const int size,
                            const int block_size,
                            const int grid_size,
                            const double* input );

#endif


template< class T > class tnlCUDAKernelsTester : public CppUnit :: TestCase
{
   public:
   tnlCUDAKernelsTester(){};

   virtual
   ~tnlCUDAKernelsTester(){};

   static CppUnit :: Test* suite()
   {
      CppUnit :: TestSuite* suiteOfTests = new CppUnit :: TestSuite( "tnlCUDAKernelsTester" );
      CppUnit :: TestResult result;
      suiteOfTests -> addTest( new CppUnit :: TestCaller< tnlCUDAKernelsTester< T > >(
                               "testReduction",
                               & tnlCUDAKernelsTester< T > :: testReduction )
                             );

      return suiteOfTests;
   };

   void testReduction()
   {
	   /*
	    * Test by Jan Vacata.
	    */
	   int size = 100;
	   int desBlockSize = 128;    //Desired block size
	   int desGridSize = 2048;    //Impose limitation on grid size so that threads could perform sequential work

	   tnlLongVector< T > host_input( size );
	   tnlLongVector< T > host_output( size );

	   tnlLongVectorCUDA< T > device_input( size );
	   tnlLongVectorCUDA< T > device_output( size );

	   for( int i=0; i < size; i ++ )
	   {
		   host_input[ i ] = 1;
		   host_output[ i ] = 0;
	   }
	   device_input. copyFrom( host_input );

	   //Calculate necessary block/grid dimensions
	   int block_size = :: Min( size/2, desBlockSize );
	   //Grid size is limited in this case
	   int grid_size = :: Min( desGridSize, size / block_size / 2 );

	   T min = tnlCUDAReductionMin( size, block_size, grid_size, device_input. Data() );
	   T max = tnlCUDAReductionMax( size, block_size, grid_size, device_input. Data() );
	   T sum = tnlCUDAReductionSum( size, block_size, grid_size, device_input. Data() );

	   cout << "Min: " << min
			<< "Max: " << max
			<< "Sum: " << sum << endl;

   }
};


#endif /* TNLCUDAKERNELSTESTER_H_ */
