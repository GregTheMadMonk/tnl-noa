/***************************************************************************
                          tnlCudaVectorOperationsTester.h  -  description
                             -------------------
    begin                : Mar 30, 2013
    copyright            : (C) 2013 by Tomas Oberhuber
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

#ifndef TNLCUDAVECTOROPERATIONSTESTER_H_
#define TNLCUDAVECTOROPERATIONSTESTER_H_

#include <tnlConfig.h>

#ifdef HAVE_CPPUNIT
#include <cppunit/TestSuite.h>
#include <cppunit/TestResult.h>
#include <cppunit/TestCaller.h>
#include <cppunit/TestCase.h>
#include <cppunit/Message.h>
#include <core/cuda/device-check.h>
#include <implementation/core/memory-operations.h>
#include <core/tnlVector.h>
#include <implementation/core/vector-operations.h>

template< typename Type >
class tnlCudaVectorOperationsTester : public CppUnit :: TestCase
{
   public:
   tnlCudaVectorOperationsTester(){};

   virtual
   ~tnlCudaVectorOperationsTester(){};

   static CppUnit :: Test* suite()
   {
      CppUnit :: TestSuite* suiteOfTests = new CppUnit :: TestSuite( "tnlCudaVectorOperationsTester" );
      CppUnit :: TestResult result;

      suiteOfTests -> addTest( new CppUnit :: TestCaller< tnlCudaVectorOperationsTester >(
                                "getVectorMaxTest",
                                &tnlCudaVectorOperationsTester :: getVectorMaxTest )
                               );
      suiteOfTests -> addTest( new CppUnit :: TestCaller< tnlCudaVectorOperationsTester >(
                                "getVectorMinTest",
                                &tnlCudaVectorOperationsTester :: getVectorMinTest )
                               );
      suiteOfTests -> addTest( new CppUnit :: TestCaller< tnlCudaVectorOperationsTester >(
                                "getVectorAbsMaxTest",
                                &tnlCudaVectorOperationsTester :: getVectorAbsMaxTest )
                               );
      suiteOfTests -> addTest( new CppUnit :: TestCaller< tnlCudaVectorOperationsTester >(
                                "getVectorAbsMinTest",
                                &tnlCudaVectorOperationsTester :: getVectorAbsMinTest )
                               );
      suiteOfTests -> addTest( new CppUnit :: TestCaller< tnlCudaVectorOperationsTester >(
                                "getVectorLpNormTest",
                                &tnlCudaVectorOperationsTester :: getVectorLpNormTest )
                               );
      suiteOfTests -> addTest( new CppUnit :: TestCaller< tnlCudaVectorOperationsTester >(
                                "getVectorSumTest",
                                &tnlCudaVectorOperationsTester :: getVectorSumTest )
                               );
      suiteOfTests -> addTest( new CppUnit :: TestCaller< tnlCudaVectorOperationsTester >(
                                "getVectorDifferenceMaxTest",
                                &tnlCudaVectorOperationsTester :: getVectorDifferenceMaxTest )
                               );
      suiteOfTests -> addTest( new CppUnit :: TestCaller< tnlCudaVectorOperationsTester >(
                                "getVectorDifferenceMinTest",
                                &tnlCudaVectorOperationsTester :: getVectorDifferenceMinTest )
                               );
      return suiteOfTests;
   };

   template< typename Vector >
   void setLinearSequence( Vector& deviceVector )
   {
      tnlVector< typename Vector :: RealType, tnlHost > a;
      a. setSize( deviceVector. getSize() );
      for( int i = 0; i < a. getSize(); i ++ )
         a. getData()[ i ] = i;

      copyMemoryHostToCuda( deviceVector. getData(),
                        a. getData(),
                        a. getSize() );
      CPPUNIT_ASSERT( checkCudaDevice );
   }


   template< typename Vector >
   void setOnesSequence( Vector& deviceVector )
   {
      tnlVector< typename Vector :: RealType, tnlHost > a;
      a. setSize( deviceVector. getSize() );
      for( int i = 0; i < a. getSize(); i ++ )
         a. getData()[ i ] = 1;

      copyMemoryHostToCuda( deviceVector. getData(),
                        a. getData(),
                        a. getSize() );
      CPPUNIT_ASSERT( checkCudaDevice );
   }


   template< typename Vector >
   void setNegativeLinearSequence( Vector& deviceVector )
   {
      tnlVector< typename Vector :: RealType, tnlHost > a;
      a. setSize( deviceVector. getSize() );
      for( int i = 0; i < a. getSize(); i ++ )
         a. getData()[ i ] = -i;

      copyMemoryHostToCuda( deviceVector. getData(),
                        a. getData(),
                        a. getSize() );
      CPPUNIT_ASSERT( checkCudaDevice );
   }


   void getVectorMaxTest()
   {
      const int size( 1234567 );
      tnlVector< Type, tnlCuda > v;
      v. setSize( size );
      setLinearSequence( v );

      CPPUNIT_ASSERT( getCudaVectorMax( v ) == size - 1 );
   }

   void getVectorMinTest()
   {
      const int size( 1234567 );
      tnlVector< Type, tnlCuda > v;
      v. setSize( size );
      setLinearSequence( v );

      CPPUNIT_ASSERT( getCudaVectorMin( v ) == 0 );
   }

   void getVectorAbsMaxTest()
   {
      const int size( 1234567 );
      tnlVector< Type, tnlCuda > v;
      v. setSize( size );
      setNegativeLinearSequence( v );

      CPPUNIT_ASSERT( getCudaVectorAbsMax( v ) == size - 1 );
   }

   void getVectorAbsMinTest()
   {
      const int size( 1234567 );
      tnlVector< Type, tnlCuda > v;
      v. setSize( size );
      setNegativeLinearSequence( v );

      CPPUNIT_ASSERT( getCudaVectorAbsMin( v ) == 0 );
   }

   void getVectorLpNormTest()
   {
      const int size( 1234567 );
      tnlVector< Type, tnlCuda > v;
      v. setSize( size );
      setOnesSequence( v );

      CPPUNIT_ASSERT( getCudaVectorLpNorm( v, 2.0 ) == sqrt( size ) );
   }

   void getVectorSumTest()
   {
      const int size( 1234567 );
      tnlVector< Type, tnlCuda > v;
      v. setSize( size );
      setOnesSequence( v );

      CPPUNIT_ASSERT( getCudaVectorSum( v ) == size );

      setLinearSequence( v );

      CPPUNIT_ASSERT( getCudaVectorSum( v ) == ( ( Type ) size ) * ( ( Type ) size - 1 ) / 2 );
   }

   void getVectorDifferenceMaxTest()
   {
      const int size( 1234567 );
      tnlVector< Type, tnlCuda > u, v;
      u. setSize( size );
      v. setSize( size );
      setLinearSequence( u );
      setOnesSequence( v );

      CPPUNIT_ASSERT( getCudaVectorDifferenceMax( u, v ) == size - 2 );
   }

   void getVectorDifferenceMinTest()
   {
      const int size( 1234567 );
      tnlVector< Type, tnlCuda > u, v;
      u. setSize( size );
      v. setSize( size );
      setLinearSequence( u );
      setOnesSequence( v );

      cout << getCudaVectorDifferenceMin( u, v ) << endl;
      CPPUNIT_ASSERT( getCudaVectorDifferenceMin( u, v ) == -1 );
   }

};

#else
template< typename Type >
class tnlCudaVectorOperationsTester
{};
#endif /* HAVE_CPPUNIT */

#endif /* TNLCUDAVECTOROPERATIONSTESTER_H_ */
