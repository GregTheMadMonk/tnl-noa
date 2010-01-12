/*
 * tnlFieldCUDA2DTester.h
 *
 *  Created on: Jan 12, 2010
 *      Author: oberhuber
 */

#ifndef TNLFIELDCUDA2DTESTER_H_

#include <cppunit/TestSuite.h>
#include <cppunit/TestResult.h>
#include <cppunit/TestCaller.h>
#include <cppunit/TestCase.h>
#include <core/tnlFieldCUDA2D.h>
#include <core/tnlField2D.h>

#ifdef HAVE_CUDA

#endif

template< class T > class tnlFieldCUDA2DTester : public CppUnit :: TestCase
{
   public:
   tnlFieldCUDA2DTester(){};

   virtual
   ~tnlFieldCUDA2DTester(){};

   static CppUnit :: Test* suite()
   {
      CppUnit :: TestSuite* suiteOfTests = new CppUnit :: TestSuite( "tnlFieldCUDA2DTester" );
      CppUnit :: TestResult result;
      suiteOfTests -> addTest( new CppUnit :: TestCaller< tnlFieldCUDA2DTester< T > >(
                               "testAllocation",
                               & tnlFieldCUDA2DTester< T > :: testAllocation )
                             );
      suiteOfTests -> addTest( new CppUnit :: TestCaller< tnlFieldCUDA2DTester< T > >(
                               "testCopying",
                               & tnlFieldCUDA2DTester< T > :: testCopying )
                              );
            return suiteOfTests;
   }

   void testCopying()
   {
#ifdef HAVE_CUDA
	   const int size = 100;
	   tnlField2D< T > host_field( size, size );
	   tnlFieldCUDA2D< T > device_field( size, size );
	   for( int i = 0; i < size; i ++ )
		   for( int j = 0; j < size; j ++ )
			   host_field( i, j ) = ( T ) ( i + j );
	   device_field. copyFrom( host_field );
	   host_field. Zeros();
	   host_field. copyFrom( device_field );
	   int errs( 0 );
	   for( int i = 0; i < size; i ++ )
		   for( int j = 0; j < size; j ++ )
			   if( host_field( i, j ) != i + j ) errs ++;
	   CPPUNIT_ASSERT( ! errs );
#endif
   }

   void testAllocation()
   {
#ifdef HAVE_CUDA
      tnlFieldCUDA2D< T > cuda_field_1;
      CPPUNIT_ASSERT( !cuda_field_1 );

      cuda_field_1. SetNewDimensions( 100, 100 );
      CPPUNIT_ASSERT( cuda_field_1 );
      CPPUNIT_ASSERT( cuda_field_1. GetXSize() == 100 );
      CPPUNIT_ASSERT( cuda_field_1. GetYSize() == 100 );

      tnlFieldCUDA2D< T > cuda_field_2;
      CPPUNIT_ASSERT( !cuda_field_2 );
      cuda_field_2. SetNewDimensions( cuda_field_1 );
      CPPUNIT_ASSERT( cuda_field_2. GetXSize() == 100 );
      CPPUNIT_ASSERT( cuda_field_2. GetYSize() == 100 );

      tnlFieldCUDA2D< T > cuda_field_3( 100, 100 );
      CPPUNIT_ASSERT( cuda_field_3. GetXSize() == 100 );
      CPPUNIT_ASSERT( cuda_field_3. GetYSize() == 100 );

      tnlFieldCUDA2D< T >* cuda_field_4 = new tnlFieldCUDA2D< T >( 100, 100 );
      tnlFieldCUDA2D< T >* cuda_field_5 = new tnlFieldCUDA2D< T >;
      CPPUNIT_ASSERT( *cuda_field_4 );
      CPPUNIT_ASSERT( ! *cuda_field_5 );

      cuda_field_5 -> SetSharedData( cuda_field_4 -> Data(),
                                     cuda_field_4 -> GetXSize(),
                                     cuda_field_4 -> GetYSize()
                                      );
      CPPUNIT_ASSERT( *cuda_field_5 );
      /* Shared data are not handled automaticaly.
       * One must be sure that data were not freed sooner
       * then any Fields stopped using it.
       */
      delete cuda_field_5;
      delete cuda_field_4;
#endif
   }

};

#define TNLFIELDCUDA2DTESTER_H_


#endif /* TNLFIELDCUDA2DTESTER_H_ */
