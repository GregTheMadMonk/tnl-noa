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
#include <math.h>
#include <cppunit/TestSuite.h>
#include <cppunit/TestResult.h>
#include <cppunit/TestCaller.h>
#include <cppunit/TestCase.h>
#include <core/tnlLongVectorCUDA.h>
#include <core/tnlLongVectorHost.h>
#include <core/mfuncs.h>

using namespace std;

#ifdef HAVE_CUDA
#include <core/tnl-cuda-kernels.h>
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

      T param;
      tnlString test_name = tnlString( "testSimpleReduction1< " ) + GetParameterType( param ) + tnlString( " >" );
      suiteOfTests -> addTest( new CppUnit :: TestCaller< tnlCUDAKernelsTester< T > >(
    		               test_name. getString(),
                               & tnlCUDAKernelsTester< T > :: testSimpleReduction1 )
                             );
      test_name = tnlString( "testSimpleReduction2< " ) + GetParameterType( param ) + tnlString( " >" );
      suiteOfTests -> addTest( new CppUnit :: TestCaller< tnlCUDAKernelsTester< T > >(
                               test_name. getString(),
                               & tnlCUDAKernelsTester< T > :: testSimpleReduction2 )
                              );
      test_name = tnlString( "testSimpleReduction3< " ) + GetParameterType( param ) + tnlString( " >" );
      suiteOfTests -> addTest( new CppUnit :: TestCaller< tnlCUDAKernelsTester< T > >(
                               test_name. getString(),
                               & tnlCUDAKernelsTester< T > :: testSimpleReduction3 )
                              );
      test_name = tnlString( "testSimpleReduction4< " ) + GetParameterType( param ) + tnlString( " >" );
      suiteOfTests -> addTest( new CppUnit :: TestCaller< tnlCUDAKernelsTester< T > >(
                               test_name. getString(),
                               & tnlCUDAKernelsTester< T > :: testSimpleReduction4 )
                              );
      test_name = tnlString( "testSimpleReduction5< " ) + GetParameterType( param ) + tnlString( " >" );
      suiteOfTests -> addTest( new CppUnit :: TestCaller< tnlCUDAKernelsTester< T > >(
                               test_name. getString(),
                               & tnlCUDAKernelsTester< T > :: testSimpleReduction5 )
                              );
      test_name = tnlString( "testReduction< " ) + GetParameterType( param ) + tnlString( " >" );
      suiteOfTests -> addTest( new CppUnit :: TestCaller< tnlCUDAKernelsTester< T > >(
                               test_name. getString(),
                               & tnlCUDAKernelsTester< T > :: testReduction )
                             );

      return suiteOfTests;
   };

   bool testSetup( tnlLongVector< T >& host_input,
		           tnlLongVector< T, tnlCuda >& device_input,
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

   bool mainReduction( const tnlLongVector< T >& host_input,
		               int algorithm_efficiency,
		               const int desired_block_size,
		               const int desired_grid_size )
   {
      const int size = host_input. getSize();
      tnlLongVector< T, tnlCuda > device_input;
      if( ! device_input. setSize( size ) )
         return false;
      device_input. copyFrom( host_input );

      T seq_min( host_input[ 0 ] ),
		 seq_max( host_input[ 0 ] ),
		 seq_sum( host_input[ 0 ] );

      for( int i = 1; i < size; i ++ )
      {
         seq_min = :: Min( seq_min, host_input[ i ] );
         seq_max = :: Max( seq_max, host_input[ i ] );
         seq_sum += host_input[ i ];
      }

      T min, max, sum;
      switch( algorithm_efficiency )
      {
         case 1:
            tnlCUDASimpleReduction1Min( size, device_input. Data(), min, 0 );
            tnlCUDASimpleReduction1Max( size, device_input. Data(), max, 0 );
            tnlCUDASimpleReduction1Sum( size, device_input. Data(), sum, 0 );
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
            tnlCUDAReductionMin( size, device_input. Data(), min );
            tnlCUDAReductionMax( size, device_input. Data(), max );
            tnlCUDAReductionSum( size, device_input. Data(), sum );
      }


      /*if( min == seq_min )
		   cout << "Min: " << min << " Seq. min: " << seq_min << " :-)" << endl;
	   else
		   cout << "Min: " << min << " Seq. min: " << seq_min << " !!!!!!!!!!" << endl;
	   if( max == seq_max )
		   cout	<< "Max: " << max << " Seq. max: " << seq_max << " :-)" << endl;
	   else
		   cout	<< "Max: " << max << " Seq. max: " << seq_max << " !!!!!!!!!!" << endl;
	   if( sum == seq_sum )
		   cout << "Sum: " << sum << " Seq. sum: " << seq_sum << " :-)" << endl;
	   else
		   cout << "Sum: " << sum << " Seq. sum: " << seq_sum << " !!!!!!!!!!" << endl;*/

      T param;
      if( GetParameterType( param ) == "float" ||
               GetParameterType( param ) == "double" )
      {
         if( min != seq_min )
            cout << "Diff. min = " << min << " seq. min = " << seq_min;
         if( max != seq_max )
            cout << "Diff. max = " << max << " seq. max = " << seq_max;
         CPPUNIT_ASSERT( min == seq_min );
         CPPUNIT_ASSERT( max == seq_max );
         if( sum == 0.0 )
         {
            CPPUNIT_ASSERT( sum == seq_sum );
         }
         else
         {
            double diff = ( ( double ) sum - ( double ) seq_sum ) / ( double) sum;
            if( fabs( diff > 1.0e-5 ) )
            {
               cout << "Diff is " << diff << " for " << GetParameterType( param ) << endl;
               abort();
            }
            CPPUNIT_ASSERT( fabs( diff ) < 1.0e-5 );
         }
      }
      else
      {
         if( min != seq_min )
         {
            cout << "Diff. min = " << min << " seq. min = " << seq_min;
            abort();
         }
         if( max != seq_max )
         {
            cout << "Diff. max = " << max << " seq. max = " << seq_max;
            abort();
         }
         if( sum != seq_sum )
         {
            cout << "Diff. sum = " << sum << " seq. sum = " << seq_sum;
            abort();
         }
         CPPUNIT_ASSERT( min == seq_min );
         CPPUNIT_ASSERT( max == seq_max );
         CPPUNIT_ASSERT( sum == seq_sum );
      }

   }

   void testReduction( int algorithm_efficiency = 0 )
   {
      tnlLongVector< T > host_input;
      int size = 2;
      for( int s = 1; s < 12; s ++ )
      {
         tnlLongVector< T > host_input( "host-input", size );

         //cout << "Alg. " << algorithm_efficiency << "Testing zeros with size "  << size << " ";
         for( int i = 0; i < size; i ++ )
            host_input[ i ] = 0.0;
         mainReduction( host_input,
                  algorithm_efficiency,
                  256,
                  2048 );

         //cout << "Alg. " << algorithm_efficiency  << "Testing ones with size "  << size << " ";
         for( int i = 0; i < size; i ++ )
            host_input[ i ] = 1.0;
         mainReduction( host_input,
                  algorithm_efficiency,
                  256,
                  2048 );
         //cout << "Alg. " << algorithm_efficiency  << "Testing linear sequence with size "  << size << " ";
         for( int i = 0; i < size; i ++ )
            host_input[ i ] = i;
         mainReduction( host_input,
                  algorithm_efficiency,
                  256,
                  2048 );
         //cout << "Alg. " << algorithm_efficiency  << "Testing quadratic sequence with size "  << size << " ";
         for( int i = 0; i < size; i ++ )
            host_input[ i ] = ( i - size / 2 ) * ( i - size / 2 );
         mainReduction( host_input,
                  algorithm_efficiency,
                  256,
                  2048 );
         size *= 2;
         //cout << endl;
      }
      for( size = 1; size < 5000; size ++ )
      {
         tnlLongVector< T > host_input( "host-input", size );

         //cout << "Alg. " << algorithm_efficiency  << " Testing zeros with size "  << size << " ";
         for( int i = 0; i < size; i ++ )
            host_input[ i ] = 0.0;
         mainReduction( host_input,
                  algorithm_efficiency,
                  256,
                  2048 );

         //cout << "Alg. " << algorithm_efficiency  << " Testing ones with size "  << size << " ";
         for( int i = 0; i < size; i ++ )
            host_input[ i ] = 1.0;
         mainReduction( host_input,
                  algorithm_efficiency,
                  256,
                  2048 );
         //cout << "Alg. " << algorithm_efficiency  << " Testing linear sequence with size "  << size << " ";
         for( int i = 0; i < size; i ++ )
            host_input[ i ] = i;
         mainReduction( host_input,
                  algorithm_efficiency,
                  256,
                  2048 );

         //cout << "Alg. " << algorithm_efficiency  << " Testing quadratic sequence with size "  << size << " ";
         for( int i = 0; i < size; i ++ )
            host_input[ i ] = ( i - size / 2 ) * ( i - size / 2 );
         mainReduction( host_input,
                  algorithm_efficiency,
                  256,
                  2048 );
         //cout << endl;
      }
   };

   void testReduction()
   {
      //cout << "Test FAST reduction" << endl;
      testReduction( 0 );
   }

   void testSimpleReduction5()
   {
      //cout << "Test reduction 5" << endl;
      testReduction( 5 );
   };

   void testSimpleReduction4()
   {
      //cout << "Test reduction 4" << endl;
      testReduction( 4 );
   };

   void testSimpleReduction3()
   {
      //cout << "Test reduction 3" << endl;
      testReduction( 3 );
   };

   void testSimpleReduction2()
   {
      //cout << "Test reduction 2" << endl;
      testReduction( 2 );
   };

   void testSimpleReduction1()
   {
      //cout << "Test reduction 1" << endl;
      testReduction( 1 );
   };


};


#endif /* TNLCUDAKERNELSTESTER_H_ */
