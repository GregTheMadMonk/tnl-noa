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

/*
 * Simple reduction 1
 */
int tnlCUDASimpleReduction1Min( const int size,
                          const int block_size,
                          const int grid_size,
                          const int* input,
                          int* output );
int tnlCUDASimpleReduction1Max( const int size,
                          const int block_size,
                          const int grid_size,
                          const int* input,
                          int* output );
int tnlCUDASimpleReduction1Sum( const int size,
                          const int block_size,
                          const int grid_size,
                          const int* input,
                          int* output );
float tnlCUDASimpleReduction1Min( const int size,
                            const int block_size,
                            const int grid_size,
                            const float* input );
float tnlCUDASimpleReduction1Max( const int size,
                            const int block_size,
                            const int grid_size,
                            const float* input );
float tnlCUDASimpleReduction1Sum( const int size,
                            const int block_size,
                            const int grid_size,
                            const float* input );
double tnlCUDASimpleReduction1Min( const int size,
                             const int block_size,
                             const int grid_size,
                             const double* input );
double tnlCUDASimpleReduction1Max( const int size,
                             const int block_size,
                             const int grid_size,
                             const double* input );
double tnlCUDASimpleReduction1Sum( const int size,
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
    		                   "testSimpleReduction1",
                               & tnlCUDAKernelsTester< T > :: testSimpleReduction1 )
                             );
      suiteOfTests -> addTest( new CppUnit :: TestCaller< tnlCUDAKernelsTester< T > >(
                               "testReduction",
                               & tnlCUDAKernelsTester< T > :: testFastReduction )
                             );

      return suiteOfTests;
   };

   bool testSetup( tnlLongVector< T >& host_input,
		           tnlLongVector< T >& host_output,
		           tnlLongVectorCUDA< T >& device_input,
		           tnlLongVectorCUDA< T >& device_output,
		           int size )
   {
	   if( ! host_input. SetNewSize( size ) )
		   return false;
	   if( ! host_output. SetNewSize( size ) )
		   return false;
	   if( ! device_input. SetNewSize( size ) )
		   return false;
	   if( ! device_output. SetNewSize( size ) )
		   return false;

	   for( int i=0; i < size; i ++ )
	   {
		   host_input[ i ] = i + 1;
		   host_output[ i ] = 0;
	   }
	   device_input. copyFrom( host_input );
	   return true;
   }

   void testReduction( int algorithm_efficiency = 0 )
   {
	   int size = 1<<16;
	   int desBlockSize = 128;    //Desired block size
	   int desGridSize = 2048;    //Impose limitation on grid size so that threads could perform sequential work

	   tnlLongVector< T > host_input, host_output;
	   tnlLongVectorCUDA< T > device_input, device_output;
	   CPPUNIT_ASSERT( testSetup( host_input,
		         	   host_output,
		         	   device_input,
		         	   device_output,
		         	   size )  );



	   //Calculate necessary block/grid dimensions
	   int block_size = :: Min( size/2, desBlockSize );
	   //Grid size is limited in this case
	   int grid_size = :: Min( desGridSize, size / block_size / 2 );

	   T min, max, sum;
	   switch( algorithm_efficiency )
	   {
		   case 1:
			   min = tnlCUDASimpleReduction1Min( size, block_size, grid_size, device_input. Data(), device_output. Data() );
			   max = tnlCUDASimpleReduction1Max( size, block_size, grid_size, device_input. Data(), device_output. Data() );
			   sum = tnlCUDASimpleReduction1Sum( size, block_size, grid_size, device_input. Data(), device_output. Data() );
			   break;
		   default:
			   min = tnlCUDAReductionMin( size, block_size, grid_size, device_input. Data() );
			   max = tnlCUDAReductionMax( size, block_size, grid_size, device_input. Data() );
			   sum = tnlCUDAReductionSum( size, block_size, grid_size, device_input. Data() );
	   }

	   cout << "Min: " << min << endl
			<< "Max: " << max << endl
			<< "Sum: " << sum << endl;

   };

   void testSimpleReduction1()
   {
	   testReduction( 1 );
   };

   void testFastReduction()
   {
	   testReduction( 0 );
   }


};


#endif /* TNLCUDAKERNELSTESTER_H_ */
