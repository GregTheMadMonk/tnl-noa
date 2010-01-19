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
 * Simple reduction 5
 */
bool tnlCUDASimpleReduction5Min( const int size,
                                 const int* input,
                                 int& result );
bool tnlCUDASimpleReduction5Max( const int size,
                                 const int* input,
                                 int& result );
bool tnlCUDASimpleReduction5Sum( const int size,
                                 const int* input,
                                 int& result );
bool tnlCUDASimpleReduction5Min( const int size,
                                 const float* input,
                                 float& result);
bool tnlCUDASimpleReduction5Max( const int size,
                                 const float* input,
                                 float& result);
bool tnlCUDASimpleReduction5Sum( const int size,
                                 const float* input,
                                 float& result);
bool tnlCUDASimpleReduction5Min( const int size,
                                 const double* input,
                                 double& result);
bool tnlCUDASimpleReduction5Max( const int size,
                                 const double* input,
                                 double& result );
bool tnlCUDASimpleReduction5Sum( const int size,
                                 const double* input,
                                 double& result );


/*
 * Simple reduction 4
 */
bool tnlCUDASimpleReduction4Min( const int size,
                          const int* input,
                          int& result );
bool tnlCUDASimpleReduction4Max( const int size,
                          const int* input,
                          int& result );
bool tnlCUDASimpleReduction4Sum( const int size,
                          const int* input,
                          int& result );
bool tnlCUDASimpleReduction4Min( const int size,
                            const float* input,
                            float& result);
bool tnlCUDASimpleReduction4Max( const int size,
                            const float* input,
                            float& result);
bool tnlCUDASimpleReduction4Sum( const int size,
                            const float* input,
                            float& result);
bool tnlCUDASimpleReduction4Min( const int size,
                             const double* input,
                             double& result);
bool tnlCUDASimpleReduction4Max( const int size,
                             const double* input,
                             double& result );
bool tnlCUDASimpleReduction4Sum( const int size,
                             const double* input,
                             double& result );

/*
 * Simple reduction 3
 */
bool tnlCUDASimpleReduction3Min( const int size,
                          const int* input,
                          int& result );
bool tnlCUDASimpleReduction3Max( const int size,
                          const int* input,
                          int& result );
bool tnlCUDASimpleReduction3Sum( const int size,
                          const int* input,
                          int& result );
bool tnlCUDASimpleReduction3Min( const int size,
                            const float* input,
                            float& result);
bool tnlCUDASimpleReduction3Max( const int size,
                            const float* input,
                            float& result);
bool tnlCUDASimpleReduction3Sum( const int size,
                            const float* input,
                            float& result);
bool tnlCUDASimpleReduction3Min( const int size,
                             const double* input,
                             double& result);
bool tnlCUDASimpleReduction3Max( const int size,
                             const double* input,
                             double& result );
bool tnlCUDASimpleReduction3Sum( const int size,
                             const double* input,
                             double& result );

/*
 * Simple reduction 2
 */
bool tnlCUDASimpleReduction2Min( const int size,
                          const int* input,
                          int& result );
bool tnlCUDASimpleReduction2Max( const int size,
                          const int* input,
                          int& result );
bool tnlCUDASimpleReduction2Sum( const int size,
                          const int* input,
                          int& result );
bool tnlCUDASimpleReduction2Min( const int size,
                            const float* input,
                            float& result);
bool tnlCUDASimpleReduction2Max( const int size,
                            const float* input,
                            float& result);
bool tnlCUDASimpleReduction2Sum( const int size,
                            const float* input,
                            float& result);
bool tnlCUDASimpleReduction2Min( const int size,
                             const double* input,
                             double& result);
bool tnlCUDASimpleReduction2Max( const int size,
                             const double* input,
                             double& result );
bool tnlCUDASimpleReduction2Sum( const int size,
                             const double* input,
                             double& result );

/*
 * Simple reduction 1
 */
bool tnlCUDASimpleReduction1Min( const int size,
                          const int* input,
                          int& result );
bool tnlCUDASimpleReduction1Max( const int size,
                          const int* input,
                          int& result );
bool tnlCUDASimpleReduction1Sum( const int size,
                          const int* input,
                          int& result );
bool tnlCUDASimpleReduction1Min( const int size,
                            const float* input,
                            float& result);
bool tnlCUDASimpleReduction1Max( const int size,
                            const float* input,
                            float& result);
bool tnlCUDASimpleReduction1Sum( const int size,
                            const float* input,
                            float& result);
bool tnlCUDASimpleReduction1Min( const int size,
                             const double* input,
                             double& result);
bool tnlCUDASimpleReduction1Max( const int size,
                             const double* input,
                             double& result );
bool tnlCUDASimpleReduction1Sum( const int size,
                             const double* input,
                             double& result );

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
        		               "testSimpleReduction2",
                               & tnlCUDAKernelsTester< T > :: testSimpleReduction2 )
                              );
      suiteOfTests -> addTest( new CppUnit :: TestCaller< tnlCUDAKernelsTester< T > >(
                      		   "testSimpleReduction3",
                               & tnlCUDAKernelsTester< T > :: testSimpleReduction3 )
                              );
      suiteOfTests -> addTest( new CppUnit :: TestCaller< tnlCUDAKernelsTester< T > >(
                               "testSimpleReduction4",
                               & tnlCUDAKernelsTester< T > :: testSimpleReduction4 )
                              );
      suiteOfTests -> addTest( new CppUnit :: TestCaller< tnlCUDAKernelsTester< T > >(
                               "testSimpleReduction5",
                               & tnlCUDAKernelsTester< T > :: testSimpleReduction5 )
                              );
      /*suiteOfTests -> addTest( new CppUnit :: TestCaller< tnlCUDAKernelsTester< T > >(
                               "testReduction",
                               & tnlCUDAKernelsTester< T > :: testFastReduction )
                             );*/

      return suiteOfTests;
   };

   bool testSetup( tnlLongVector< T >& host_input,
		           tnlLongVectorCUDA< T >& device_input,
		           int size )
   {
	   if( ! host_input. SetNewSize( size ) )
		   return false;
	   if( ! device_input. SetNewSize( size ) )
		   return false;

	   for( int i=0; i < size; i ++ )
		   host_input[ i ] = i + 1;

	   device_input. copyFrom( host_input );
	   return true;
   }

   void testReduction( int algorithm_efficiency = 0 )
   {
	   int size = 1<<10;
	   int desBlockSize = 128;    //Desired block size
	   int desGridSize = 2048;    //Impose limitation on grid size so that threads could perform sequential work

	   tnlLongVector< T > host_input;
	   tnlLongVectorCUDA< T > device_input;
	   CPPUNIT_ASSERT( testSetup( host_input,
		         	   device_input,
		         	   size )  );
	   T seq_min( host_input[ 0 ] ),
	     seq_max( host_input[ 0 ] ),
	     seq_sum( host_input[ 0 ] );
	   for( int i = 1; i < size; i ++ )
	   {
		   seq_min = :: Min( seq_min, host_input[ i ] );
		   seq_max = :: Max( seq_max, host_input[ i ] );
		   seq_sum += host_input[ i ];
	   }

	   //Calculate necessary block/grid dimensions
	   int block_size = :: Min( size/2, desBlockSize );
	   //Grid size is limited in this case
	   int grid_size = :: Min( desGridSize, size / block_size / 2 );

	   T min, max, sum;
	   switch( algorithm_efficiency )
	   {
		   case 1:
			   tnlCUDASimpleReduction1Min( size, device_input. Data(), min );
			   tnlCUDASimpleReduction1Max( size, device_input. Data(), max );
			   tnlCUDASimpleReduction1Sum( size, device_input. Data(), sum );
			   break;
		   case 2:
			   tnlCUDASimpleReduction2Min( size, device_input. Data(), min );
			   tnlCUDASimpleReduction2Max( size, device_input. Data(), max );
			   tnlCUDASimpleReduction2Sum( size, device_input. Data(), sum );
			   break;
		   case 3:
			   tnlCUDASimpleReduction3Min( size, device_input. Data(), min );
			   tnlCUDASimpleReduction3Max( size, device_input. Data(), max );
			   tnlCUDASimpleReduction3Sum( size, device_input. Data(), sum );
			   break;
		   case 4:
			   tnlCUDASimpleReduction4Min( size, device_input. Data(), min );
			   tnlCUDASimpleReduction4Max( size, device_input. Data(), max );
			   tnlCUDASimpleReduction4Sum( size, device_input. Data(), sum );
			   break;
		   case 5:
			   tnlCUDASimpleReduction5Min( size, device_input. Data(), min );
			   tnlCUDASimpleReduction5Max( size, device_input. Data(), max );
			   tnlCUDASimpleReduction5Sum( size, device_input. Data(), sum );
			   break;
		   default:
			   min = tnlCUDAReductionMin( size, block_size, grid_size, device_input. Data() );
			   max = tnlCUDAReductionMax( size, block_size, grid_size, device_input. Data() );
			   sum = tnlCUDAReductionSum( size, block_size, grid_size, device_input. Data() );
	   }


	   cout << "Min: " << min << " Seq. min: " << seq_min << endl
			<< "Max: " << max << " Seq. max: " << seq_max << endl
			<< "Sum: " << sum << " Seq. sum: " << seq_sum << endl;

	   CPPUNIT_ASSERT( min == seq_min );
	   CPPUNIT_ASSERT( max == seq_max );
	   CPPUNIT_ASSERT( sum == seq_sum );

   };

   void testSimpleReduction5()
   {
   	   cout << "Test reduction 5" << endl;
   	   testReduction( 5 );
   };

   void testSimpleReduction4()
   {
	   cout << "Test reduction 4" << endl;
	   testReduction( 4 );
   };

   void testSimpleReduction3()
   {
   	   cout << "Test reduction 3" << endl;
     	   testReduction( 3 );
   };

   void testSimpleReduction2()
   {
	   cout << "Test reduction 2" << endl;
  	   testReduction( 2 );
   };

   void testSimpleReduction1()
   {
	   cout << "Test reduction 1" << endl;
	   testReduction( 1 );
   };

   void testFastReduction()
   {
	   testReduction( 0 );
   }


};


#endif /* TNLCUDAKERNELSTESTER_H_ */
