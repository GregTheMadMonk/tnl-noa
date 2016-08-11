/***************************************************************************
                          Devices::CudaKernelsTester.h
                             -------------------
    begin                : Jan 14, 2010
    copyright            : (C) 2009 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#ifndef Devices::CudaKERNELSTESTER_H_
#define Devices::CudaKERNELSTESTER_H_

#include <iostream>
#include <math.h>
#include <cppunit/TestSuite.h>
#include <cppunit/TestResult.h>
#include <cppunit/TestCaller.h>
#include <cppunit/TestCase.h>
#include <TNL/Containers/VectorCUDA.h>
#include <TNL/Containers/VectorHost.h>
#include <TNL/core/mfuncs.h>

#ifdef HAVE_CUDA
#include <TNL/core/tnl-cuda-kernels.h>
#endif

using namespace std;
using namespace TNL;


template< class T > class Devices::CudaKernelsTester : public CppUnit :: TestCase
{
   public:
   Devices::CudaKernelsTester(){};

   virtual
   ~Devices::CudaKernelsTester(){};

   static CppUnit :: Test* suite()
   {
      CppUnit :: TestSuite* suiteOfTests = new CppUnit :: TestSuite( "Devices::CudaKernelsTester" );
      CppUnit :: TestResult result;

      T param;
      String test_name = String( "testSimpleReduction1< " ) + getType( param ) + String( " >" );
      suiteOfTests -> addTest( new CppUnit :: TestCaller< Devices::CudaKernelsTester< T > >(
    		               test_name. getString(),
                               & Devices::CudaKernelsTester< T > :: testSimpleReduction1 )
                             );
      test_name = String( "testSimpleReduction2< " ) + getType( param ) + String( " >" );
      suiteOfTests -> addTest( new CppUnit :: TestCaller< Devices::CudaKernelsTester< T > >(
                               test_name. getString(),
                               & Devices::CudaKernelsTester< T > :: testSimpleReduction2 )
                              );
      test_name = String( "testSimpleReduction3< " ) + getType( param ) + String( " >" );
      suiteOfTests -> addTest( new CppUnit :: TestCaller< Devices::CudaKernelsTester< T > >(
                               test_name. getString(),
                               & Devices::CudaKernelsTester< T > :: testSimpleReduction3 )
                              );
      test_name = String( "testSimpleReduction4< " ) + getType( param ) + String( " >" );
      suiteOfTests -> addTest( new CppUnit :: TestCaller< Devices::CudaKernelsTester< T > >(
                               test_name. getString(),
                               & Devices::CudaKernelsTester< T > :: testSimpleReduction4 )
                              );
      test_name = String( "testSimpleReduction5< " ) + getType( param ) + String( " >" );
      suiteOfTests -> addTest( new CppUnit :: TestCaller< Devices::CudaKernelsTester< T > >(
                               test_name. getString(),
                               & Devices::CudaKernelsTester< T > :: testSimpleReduction5 )
                              );
      test_name = String( "testReduction< " ) + getType( param ) + String( " >" );
      suiteOfTests -> addTest( new CppUnit :: TestCaller< Devices::CudaKernelsTester< T > >(
                               test_name. getString(),
                               & Devices::CudaKernelsTester< T > :: testReduction )
                             );

      return suiteOfTests;
   };

   bool testSetup( Vector< T >& host_input,
		           Vector< T, Devices::Cuda >& device_input,
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

   bool mainReduction( const Vector< T >& host_input,
		               int algorithm_efficiency,
		               const int desired_block_size,
		               const int desired_grid_size )
   {
      const int size = host_input. getSize();
      Vector< T, Devices::Cuda > device_input;
      if( ! device_input. setSize( size ) )
         return false;
      device_input. copyFrom( host_input );

      T seq_min( host_input[ 0 ] ),
		 seq_max( host_input[ 0 ] ),
		 seq_sum( host_input[ 0 ] );

      for( int i = 1; i < size; i ++ )
      {
         seq_min = :: min( seq_min, host_input[ i ] );
         seq_max = :: max( seq_max, host_input[ i ] );
         seq_sum += host_input[ i ];
      }

      T min, max, sum;
      switch( algorithm_efficiency )
      {
         case 1:
            Devices::CudaSimpleReduction1Min( size, device_input. Data(), min, 0 );
            Devices::CudaSimpleReduction1Max( size, device_input. Data(), max, 0 );
            Devices::CudaSimpleReduction1Sum( size, device_input. Data(), sum, 0 );
            break;
         case 2:
            Devices::CudaSimpleReduction2Min( size, device_input. Data(), min );
            Devices::CudaSimpleReduction2Max( size, device_input. Data(), max );
            Devices::CudaSimpleReduction2Sum( size, device_input. Data(), sum );
            break;
         case 3:
            Devices::CudaSimpleReduction3Min( size, device_input. Data(), min );
            Devices::CudaSimpleReduction3Max( size, device_input. Data(), max );
            Devices::CudaSimpleReduction3Sum( size, device_input. Data(), sum );
            break;
         case 4:
            Devices::CudaSimpleReduction4Min( size, device_input. Data(), min );
            Devices::CudaSimpleReduction4Max( size, device_input. Data(), max );
            Devices::CudaSimpleReduction4Sum( size, device_input. Data(), sum );
            break;
         case 5:
            Devices::CudaSimpleReduction5Min( size, device_input. Data(), min );
            Devices::CudaSimpleReduction5Max( size, device_input. Data(), max );
            Devices::CudaSimpleReduction5Sum( size, device_input. Data(), sum );
            break;
         default:
            tnlCudaReductionMin( size, device_input. Data(), min );
            tnlCudaReductionMax( size, device_input. Data(), max );
            tnlCudaReductionSum( size, device_input. Data(), sum );
      }


      /*if( min == seq_min )
		  std::cout << "Min: " << min << " Seq. min: " << seq_min << " :-)" << std::endl;
	   else
		  std::cout << "Min: " << min << " Seq. min: " << seq_min << " !!!!!!!!!!" << std::endl;
	   if( max == seq_max )
		  std::cout	<< "Max: " << max << " Seq. max: " << seq_max << " :-)" << std::endl;
	   else
		  std::cout	<< "Max: " << max << " Seq. max: " << seq_max << " !!!!!!!!!!" << std::endl;
	   if( sum == seq_sum )
		  std::cout << "Sum: " << sum << " Seq. sum: " << seq_sum << " :-)" << std::endl;
	   else
		  std::cout << "Sum: " << sum << " Seq. sum: " << seq_sum << " !!!!!!!!!!" << std::endl;*/

      T param;
      if( getType( param ) == "float" ||
               getType( param ) == "double" )
      {
         if( min != seq_min )
           std::cout << "Diff. min = " << min << " seq. min = " << seq_min;
         if( max != seq_max )
           std::cout << "Diff. max = " << max << " seq. max = " << seq_max;
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
              std::cout << "Diff is " << diff << " for " << getType( param ) << std::endl;
               abort();
            }
            CPPUNIT_ASSERT( fabs( diff ) < 1.0e-5 );
         }
      }
      else
      {
         if( min != seq_min )
         {
           std::cout << "Diff. min = " << min << " seq. min = " << seq_min;
            abort();
         }
         if( max != seq_max )
         {
           std::cout << "Diff. max = " << max << " seq. max = " << seq_max;
            abort();
         }
         if( sum != seq_sum )
         {
           std::cout << "Diff. sum = " << sum << " seq. sum = " << seq_sum;
            abort();
         }
         CPPUNIT_ASSERT( min == seq_min );
         CPPUNIT_ASSERT( max == seq_max );
         CPPUNIT_ASSERT( sum == seq_sum );
      }

   }

   void testReduction( int algorithm_efficiency = 0 )
   {
      Vector< T > host_input;
      int size = 2;
      for( int s = 1; s < 12; s ++ )
      {
         Vector< T > host_input( "host-input", size );

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
         //cout << std::endl;
      }
      for( size = 1; size < 5000; size ++ )
      {
         Vector< T > host_input( "host-input", size );

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
         //cout << std::endl;
      }
   };

   void testReduction()
   {
      //cout << "Test FAST reduction" << std::endl;
      testReduction( 0 );
   }

   void testSimpleReduction5()
   {
      //cout << "Test reduction 5" << std::endl;
      testReduction( 5 );
   };

   void testSimpleReduction4()
   {
      //cout << "Test reduction 4" << std::endl;
      testReduction( 4 );
   };

   void testSimpleReduction3()
   {
      //cout << "Test reduction 3" << std::endl;
      testReduction( 3 );
   };

   void testSimpleReduction2()
   {
      //cout << "Test reduction 2" << std::endl;
      testReduction( 2 );
   };

   void testSimpleReduction1()
   {
      //cout << "Test reduction 1" << std::endl;
      testReduction( 1 );
   };


};


#endif /* Devices::CudaKERNELSTESTER_H_ */
