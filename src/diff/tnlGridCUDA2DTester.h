/*
 * tnlGridCUDA2DTester.h
 *
 *  Created on: Jan 13, 2010
 *      Author: Tomas Oberhuber
 */

#ifndef TNLGRIDCUDA2DTESTER_H_

#include <cppunit/TestSuite.h>
#include <cppunit/TestResult.h>
#include <cppunit/TestCaller.h>
#include <cppunit/TestCase.h>
#include <diff/tnlGridCUDA2D.h>
#include <diff/tnlGrid2D.h>

#ifdef HAVE_CUDA

#endif

template< class T > class tnlGridCUDA2DTester : public CppUnit :: TestCase
{
   public:
   tnlGridCUDA2DTester(){};

   virtual
   ~tnlGridCUDA2DTester(){};

   static CppUnit :: Test* suite()
   {
      CppUnit :: TestSuite* suiteOfTests = new CppUnit :: TestSuite( "tnlGridCUDA2DTester" );
      CppUnit :: TestResult result;
      suiteOfTests -> addTest( new CppUnit :: TestCaller< tnlGridCUDA2DTester< T > >(
                               "testAllocation",
                               & tnlGridCUDA2DTester< T > :: testAllocation )
                             );
      suiteOfTests -> addTest( new CppUnit :: TestCaller< tnlGridCUDA2DTester< T > >(
                               "testCopying",
                               & tnlGridCUDA2DTester< T > :: testCopying )
                              );
            return suiteOfTests;
   }

   void testCopying()
   {
#ifdef HAVE_CUDA
	   const int size = 100;
	   tnlGrid2D< T > host_grid( "host-grid", size, size, 0.0, 1.0, 0.0, 1.0 );
	   tnlGridCUDA2D< T > device_grid( "device-grid", size, size, 0.0, 1.0, 0.0, 1.0 );
	   for( int i = 0; i < size; i ++ )
		   for( int j = 0; j < size; j ++ )
			   host_grid( i, j ) = ( T ) ( i + j );
	   device_grid. copyFrom( host_grid );
	   host_grid. Zeros();
	   host_grid. copyFrom( device_grid );
	   int errs( 0 );
	   for( int i = 0; i < size; i ++ )
		   for( int j = 0; j < size; j ++ )
			   if( host_grid( i, j ) != i + j ) errs ++;
	   CPPUNIT_ASSERT( ! errs );
#endif
   }

   void testAllocation()
   {
#ifdef HAVE_CUDA
      tnlGridCUDA2D< T > cuda_grid_1;
      CPPUNIT_ASSERT( !cuda_grid_1 );

      cuda_grid_1. SetNewDimensions( 100, 100 );
      cuda_grid_1. SetNewDomain( 0.0, 1.0, 0.0, 1.0 );
      CPPUNIT_ASSERT( cuda_grid_1 );
      CPPUNIT_ASSERT( cuda_grid_1. GetXSize() == 100 );
      CPPUNIT_ASSERT( cuda_grid_1. GetYSize() == 100 );
      CPPUNIT_ASSERT( cuda_grid_1. GetAx() == 0.0 );
      CPPUNIT_ASSERT( cuda_grid_1. GetBx() == 1.0 );
      CPPUNIT_ASSERT( cuda_grid_1. GetAy() == 0.0 );
      CPPUNIT_ASSERT( cuda_grid_1. GetBy() == 1.0 );

      tnlGridCUDA2D< T > cuda_grid_2;
      CPPUNIT_ASSERT( !cuda_grid_2 );
      cuda_grid_2. SetNewDimensions( cuda_grid_1 );
      cuda_grid_2. SetNewDomain( cuda_grid_1 );
      CPPUNIT_ASSERT( cuda_grid_2. GetXSize() == 100 );
      CPPUNIT_ASSERT( cuda_grid_2. GetYSize() == 100 );
      CPPUNIT_ASSERT( cuda_grid_2. GetAx() == 0.0 );
      CPPUNIT_ASSERT( cuda_grid_2. GetBx() == 1.0 );
      CPPUNIT_ASSERT( cuda_grid_2. GetAy() == 0.0 );
      CPPUNIT_ASSERT( cuda_grid_2. GetBy() == 1.0 );

      tnlGridCUDA2D< T > cuda_grid_3( "cuda-grid-3", 100, 100, 0.0, 1.0, 0.0, 1.0 );
      CPPUNIT_ASSERT( cuda_grid_3. GetXSize() == 100 );
      CPPUNIT_ASSERT( cuda_grid_3. GetYSize() == 100 );
      CPPUNIT_ASSERT( cuda_grid_3. GetAx() == 0.0 );
      CPPUNIT_ASSERT( cuda_grid_3. GetBx() == 1.0 );
      CPPUNIT_ASSERT( cuda_grid_3. GetAy() == 0.0 );
      CPPUNIT_ASSERT( cuda_grid_3. GetBy() == 1.0 );

      tnlGridCUDA2D< T >* cuda_grid_4 = new tnlGridCUDA2D< T >( "cuda-grid-4", 100, 100, 0.0, 1.0, 0.0, 1.0 );
      tnlGridCUDA2D< T >* cuda_grid_5 = new tnlGridCUDA2D< T >;
      CPPUNIT_ASSERT( *cuda_grid_4 );
      CPPUNIT_ASSERT( ! *cuda_grid_5 );

      cuda_grid_5 -> SetSharedData( cuda_grid_4 -> Data(),
                                    cuda_grid_4 -> GetXSize(),
                                    cuda_grid_4 -> GetYSize()
                                   );
      CPPUNIT_ASSERT( *cuda_grid_5 );
      /* Shared data are not handled automaticaly.
       * One must be sure that data were not freed sooner
       * then any Grids stopped using it.
       */
      delete cuda_grid_5;
      delete cuda_grid_4;
#endif
   }

};

#define TNLGRIDCUDA2DTESTER_H_


#endif /* TNLGRIDCUDA2DTESTER_H_ */
