/***************************************************************************
                          tnlVectorOperationsTester.h  -  description
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

#ifndef TNLVECTOROPERATIONSTESTER_H_
#define TNLVECTOROPERATIONSTESTER_H_

#include <tnlConfig.h>

#ifdef HAVE_CPPUNIT
#include <cppunit/TestSuite.h>
#include <cppunit/TestResult.h>
#include <cppunit/TestCaller.h>
#include <cppunit/TestCase.h>
#include <cppunit/Message.h>

#include <core/vectors/tnlVector.h>
#include <core/vectors/tnlVectorOperations.h>

template< typename Real, typename Device >
class tnlVectorOperationsTester : public CppUnit :: TestCase
{
   public:
   tnlVectorOperationsTester(){};

   virtual
   ~tnlVectorOperationsTester(){};

   static CppUnit :: Test* suite()
   {
      CppUnit :: TestSuite* suiteOfTests = new CppUnit :: TestSuite( "tnlVectorOperationsTester" );
      CppUnit :: TestResult result;

      suiteOfTests -> addTest( new CppUnit :: TestCaller< tnlVectorOperationsTester >(
                                "getVectorMaxTest",
                                &tnlVectorOperationsTester :: getVectorMaxTest )
                               );
      suiteOfTests -> addTest( new CppUnit :: TestCaller< tnlVectorOperationsTester >(
                                "getVectorMinTest",
                                &tnlVectorOperationsTester :: getVectorMinTest )
                               );
      suiteOfTests -> addTest( new CppUnit :: TestCaller< tnlVectorOperationsTester >(
                                "getVectorAbsMaxTest",
                                &tnlVectorOperationsTester :: getVectorAbsMaxTest )
                               );
      suiteOfTests -> addTest( new CppUnit :: TestCaller< tnlVectorOperationsTester >(
                                "getVectorAbsMinTest",
                                &tnlVectorOperationsTester :: getVectorAbsMinTest )
                               );
      suiteOfTests -> addTest( new CppUnit :: TestCaller< tnlVectorOperationsTester >(
                                "getVectorLpNormTest",
                                &tnlVectorOperationsTester :: getVectorLpNormTest )
                               );
      suiteOfTests -> addTest( new CppUnit :: TestCaller< tnlVectorOperationsTester >(
                                "getVectorSumTest",
                                &tnlVectorOperationsTester :: getVectorSumTest )
                               );
      suiteOfTests -> addTest( new CppUnit :: TestCaller< tnlVectorOperationsTester >(
                                "getVectorDifferenceMaxTest",
                                &tnlVectorOperationsTester :: getVectorDifferenceMaxTest )
                               );
      suiteOfTests -> addTest( new CppUnit :: TestCaller< tnlVectorOperationsTester >(
                                "getVectorDifferenceMinTest",
                                &tnlVectorOperationsTester :: getVectorDifferenceMinTest )
                               );
      suiteOfTests -> addTest( new CppUnit :: TestCaller< tnlVectorOperationsTester >(
                                "getVectorDifferenceAbsMaxTest",
                                &tnlVectorOperationsTester :: getVectorDifferenceAbsMaxTest )
                               );
      suiteOfTests -> addTest( new CppUnit :: TestCaller< tnlVectorOperationsTester >(
                                "getVectorDifferenceAbsMinTest",
                                &tnlVectorOperationsTester :: getVectorDifferenceAbsMinTest )
                               );
      suiteOfTests -> addTest( new CppUnit :: TestCaller< tnlVectorOperationsTester >(
                                "getVectorDifferenceLpNormTest",
                                &tnlVectorOperationsTester :: getVectorDifferenceLpNormTest )
                               );
      suiteOfTests -> addTest( new CppUnit :: TestCaller< tnlVectorOperationsTester >(
                                "getVectorDifferenceSumTest",
                                &tnlVectorOperationsTester :: getVectorDifferenceSumTest )
                               );
      suiteOfTests -> addTest( new CppUnit :: TestCaller< tnlVectorOperationsTester >(
                                "vectorScalarMultiplicationTest",
                                &tnlVectorOperationsTester :: vectorScalarMultiplicationTest )
                               );
      suiteOfTests -> addTest( new CppUnit :: TestCaller< tnlVectorOperationsTester >(
                                "getSclaraProductTest",
                                &tnlVectorOperationsTester :: getVectorScalarProductTest )
                               );
      suiteOfTests -> addTest( new CppUnit :: TestCaller< tnlVectorOperationsTester >(
                                "alphaXPlusYTest",
                                &tnlVectorOperationsTester :: alphaXPlusYTest )
                               );
      suiteOfTests -> addTest( new CppUnit :: TestCaller< tnlVectorOperationsTester >(
                                "alphaXPlusBetaYTest",
                                &tnlVectorOperationsTester :: alphaXPlusBetaYTest )
                               );
      suiteOfTests -> addTest( new CppUnit :: TestCaller< tnlVectorOperationsTester >(
                                "alphaXPlusBetaZTest",
                                &tnlVectorOperationsTester :: alphaXPlusBetaZTest )
                               );
      suiteOfTests -> addTest( new CppUnit :: TestCaller< tnlVectorOperationsTester >(
                                "alphaXPlusBetaZPlusYTest",
                                &tnlVectorOperationsTester :: alphaXPlusBetaZPlusYTest )
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

      tnlArrayOperations< typename Vector::DeviceType,
                          tnlHost >::
      template copyMemory< typename Vector::RealType,
                           typename Vector::RealType,
                           typename Vector::IndexType >
                         ( deviceVector.getData(),
                           a.getData(),
                           a.getSize() );
   }


   template< typename Vector >
   void setOnesSequence( Vector& deviceVector )
   {
      tnlVector< typename Vector :: RealType, tnlHost > a;
      a. setSize( deviceVector. getSize() );
      for( int i = 0; i < a. getSize(); i ++ )
         a. getData()[ i ] = 1;

      tnlArrayOperations< typename Vector::DeviceType,
                          tnlHost >::
      template copyMemory< typename Vector::RealType,
                           typename Vector::RealType,
                           typename Vector::IndexType >
                         ( deviceVector.getData(),
                           a.getData(),
                           a.getSize() );
   }


   template< typename Vector >
   void setNegativeLinearSequence( Vector& deviceVector )
   {
      tnlVector< typename Vector :: RealType, tnlHost > a;
      a. setSize( deviceVector. getSize() );
      for( int i = 0; i < a. getSize(); i ++ )
         a. getData()[ i ] = -i;

      tnlArrayOperations< typename Vector::DeviceType,
                          tnlHost >::
      template copyMemory< typename Vector::RealType,
                           typename Vector::RealType,
                           typename Vector::IndexType >
                         ( deviceVector.getData(),
                           a.getData(),
                           a.getSize() );
   }

   template< typename Vector >
   void setOscilatingSequence( Vector& deviceVector,
                               typename Vector::RealType v )
   {
      tnlVector< typename Vector::RealType, tnlHost > a;
      a.setSize( deviceVector. getSize() );
      a[ 0 ] = v;
      for( int i = 1; i < a. getSize(); i ++ )
         a.getData()[ i ] = a.getData()[ i-1 ] * -1;

      tnlArrayOperations< typename Vector::DeviceType,
                          tnlHost >::
      template copyMemory< typename Vector::RealType,
                           typename Vector::RealType,
                           typename Vector::IndexType >
                         ( deviceVector.getData(),
                           a.getData(),
                           a.getSize() );
   }


   void getVectorMaxTest()
   {
      const int size( 1234567 );
      tnlVector< Real, Device > v;
      v. setSize( size );
      setLinearSequence( v );

      CPPUNIT_ASSERT( tnlVectorOperations< Device > :: getVectorMax( v.getData(), size ) == size - 1 );      
   }

   void getVectorMinTest()
   {
      const int size( 1234567 );
      tnlVector< Real, Device > v;
      v. setSize( size );
      setLinearSequence( v );

      CPPUNIT_ASSERT( tnlVectorOperations< Device > :: getVectorMin( v ) == 0 );
   }

   void getVectorAbsMaxTest()
   {
      const int size( 1234567 );
      tnlVector< Real, Device > v;
      v. setSize( size );
      setNegativeLinearSequence( v );

      CPPUNIT_ASSERT( tnlVectorOperations< Device > :: getVectorAbsMax( v ) == size - 1 );
   }

   void getVectorAbsMinTest()
   {
      const int size( 1234567 );
      tnlVector< Real, Device > v;
      v. setSize( size );
      setNegativeLinearSequence( v );

      CPPUNIT_ASSERT( tnlVectorOperations< Device > :: getVectorAbsMin( v ) == 0 );
   }

   void getVectorLpNormTest()
   {
      const int size( 1234567 );
      tnlVector< Real, Device > v;
      v. setSize( size );
      setOnesSequence( v );

      CPPUNIT_ASSERT( tnlVectorOperations< Device > :: getVectorLpNorm( v, 2.0 ) == sqrt( size ) );
   }

   void getVectorSumTest()
   {
      const int size( 1234567 );
      tnlVector< Real, Device > v;
      v. setSize( size );
      setOnesSequence( v );

      CPPUNIT_ASSERT( tnlVectorOperations< Device > :: getVectorSum( v ) == size );

      setLinearSequence( v );

      CPPUNIT_ASSERT( tnlVectorOperations< Device > :: getVectorSum( v ) == ( ( Real ) size ) * ( ( Real ) size - 1 ) / 2 );
   }

   void getVectorDifferenceMaxTest()
   {
      const int size( 1234567 );
      tnlVector< Real, Device > u, v;
      u. setSize( size );
      v. setSize( size );
      setLinearSequence( u );
      setOnesSequence( v );

      CPPUNIT_ASSERT( tnlVectorOperations< Device > :: getVectorDifferenceMax( u, v ) == size - 2 );
   }

   void getVectorDifferenceMinTest()
   {
      const int size( 1234567 );
      tnlVector< Real, Device > u, v;
      u. setSize( size );
      v. setSize( size );
      setLinearSequence( u );
      setOnesSequence( v );

      CPPUNIT_ASSERT( tnlVectorOperations< Device > :: getVectorDifferenceMin( u, v ) == -1 );
      CPPUNIT_ASSERT( tnlVectorOperations< Device > :: getVectorDifferenceMin( v, u ) == -1234565 );
   }

   void getVectorDifferenceAbsMaxTest()
   {
      const int size( 1234567 );
      tnlVector< Real, Device > u, v;
      u. setSize( size );
      v. setSize( size );
      setNegativeLinearSequence( u );
      setOnesSequence( v );

      CPPUNIT_ASSERT( tnlVectorOperations< Device > :: getVectorDifferenceAbsMax( u, v ) == size );
   }

   void getVectorDifferenceAbsMinTest()
   {
      const int size( 1234567 );
      tnlVector< Real, Device > u, v;
      u. setSize( size );
      v. setSize( size );
      setLinearSequence( u );
      setOnesSequence( v );

      CPPUNIT_ASSERT( tnlVectorOperations< Device > :: getVectorDifferenceAbsMin( u, v ) == 0 );
      CPPUNIT_ASSERT( tnlVectorOperations< Device > :: getVectorDifferenceAbsMin( v, u ) == 0 );
   }

   void getVectorDifferenceLpNormTest()
   {
      const int size( 1024 );
      tnlVector< Real, Device > u, v;
      u. setSize( size );
      v. setSize( size );
      u. setValue( 3.0 );
      v. setValue( 1.0 );

      CPPUNIT_ASSERT( tnlVectorOperations< Device > :: getVectorDifferenceLpNorm( u, v, 1.0 ) == 2.0 * size );
      CPPUNIT_ASSERT( tnlVectorOperations< Device > :: getVectorDifferenceLpNorm( u, v, 2.0 ) == sqrt( 4.0 * size ) );
   }

   void getVectorDifferenceSumTest()
   {
      const int size( 1024 );
      tnlVector< Real, Device > u, v;
      u. setSize( size );
      v. setSize( size );
      u. setValue( 3.0 );
      v. setValue( 1.0 );

      CPPUNIT_ASSERT( tnlVectorOperations< Device > :: getVectorDifferenceSum( u, v ) == 2.0 * size );
   }

   void vectorScalarMultiplicationTest()
   {
      const int size( 1025 );
      tnlVector< Real, Device > u;
      u. setSize( size );
      setLinearSequence( u );

      tnlVectorOperations< Device >::vectorScalarMultiplication( u, 3.0 );

      for( int i = 0; i < size; i++ )
         CPPUNIT_ASSERT( u.getElement( i ) == 3.0 * i );
   }

   void getVectorScalarProductTest()
   {
      const int size( 1025 );
      tnlVector< Real, Device > u, v;
      u. setSize( size );
      v. setSize( size );
      setOscilatingSequence( u, 1.0 );
      setOnesSequence( v );

      CPPUNIT_ASSERT( tnlVectorOperations< Device > :: getScalarProduct( u, v ) == 1.0 );
   }

   void alphaXPlusYTest()
   {
      const int size( 65536 );
      tnlVector< Real, Device > x, y;
      x.setSize( size );
      y.setSize( size );
      setLinearSequence( x );
      setOnesSequence( y );
      tnlVectorOperations< Device >:: alphaXPlusY( y, x, 3.0 );

      for( int i = 0; i < size; i ++ )
         CPPUNIT_ASSERT( y.getElement( i ) == 1.0 + 3.0 * i );
   };

   void alphaXPlusBetaYTest()
   {
      const int size( 65536 );
      tnlVector< Real, Device > x, y;
      x.setSize( size );
      y.setSize( size );
      setLinearSequence( x );
      setOnesSequence( y );
      tnlVectorOperations< Device >:: alphaXPlusBetaY( y, x, 3.0, -2.0 );

      for( int i = 0; i < size; i ++ )
         CPPUNIT_ASSERT( y.getElement( i ) == -2.0 + 3.0 * i );
   };

   void alphaXPlusBetaZTest()
   {
      const int size( 65536 );
      tnlVector< Real, Device > x, y, z;
      x.setSize( size );
      y.setSize( size );
      z.setSize( size );
      setLinearSequence( x );
      setOnesSequence( z );
      tnlVectorOperations< Device >:: alphaXPlusBetaZ( y, x, 3.0, z, -2.0 );

      for( int i = 0; i < size; i ++ )
         CPPUNIT_ASSERT( y.getElement( i ) == -2.0 + 3.0 * i );
   };

   void alphaXPlusBetaZPlusYTest()
   {
      const int size( 65536 );
      tnlVector< Real, Device > x, y, z;
      x.setSize( size );
      y.setSize( size );
      z.setSize( size );
      setLinearSequence( x );
      setOnesSequence( z );
      setOnesSequence( y );
      tnlVectorOperations< Device >:: alphaXPlusBetaZPlusY( y, x, 3.0, z, -2.0 );

      for( int i = 0; i < size; i ++ )
         CPPUNIT_ASSERT( y.getElement( i ) == -1.0 + 3.0 * i );
   };
};

#else
template< typename Real, typename Device >
class tnlVectorOperationsTester
{};
#endif /* HAVE_CPPUNIT */

#endif /* TNLVECTOROPERATIONSTESTER_H_ */
