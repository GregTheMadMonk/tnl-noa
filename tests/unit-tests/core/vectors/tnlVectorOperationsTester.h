/***************************************************************************
                          VectorOperationsTester.h  -  description
                             -------------------
    begin                : Mar 30, 2013
    copyright            : (C) 2013 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#ifndef VectorOPERATIONSTESTER_H_
#define VectorOPERATIONSTESTER_H_

#include <TNL/tnlConfig.h>

#ifdef HAVE_CPPUNIT
#include <cppunit/TestSuite.h>
#include <cppunit/TestResult.h>
#include <cppunit/TestCaller.h>
#include <cppunit/TestCase.h>
#include <cppunit/Message.h>

#include <TNL/Vectors/Vector.h>
#include <TNL/Vectors/VectorOperations.h>

using namespace TNL;
using namespace TNL::Arrays;

template< typename Real, typename Device >
class VectorOperationsTester : public CppUnit :: TestCase
{
   public:

   typedef CppUnit::TestCaller< VectorOperationsTester< Real, Device > > TestCallerType;

   VectorOperationsTester(){};

   virtual
   ~VectorOperationsTester(){};

   static CppUnit :: Test* suite()
   {
      CppUnit :: TestSuite* suiteOfTests = new CppUnit :: TestSuite( "VectorOperationsTester" );
      CppUnit :: TestResult result;

      suiteOfTests -> addTest( new TestCallerType( "getVectorMaxTest", &VectorOperationsTester::getVectorMaxTest ) );
      suiteOfTests -> addTest( new TestCallerType( "getVectorMinTest", &VectorOperationsTester::getVectorMinTest ) );
      suiteOfTests -> addTest( new TestCallerType( "getVectorAbsMaxTest", &VectorOperationsTester::getVectorAbsMaxTest ) );
      suiteOfTests -> addTest( new TestCallerType( "getVectorAbsMinTest", &VectorOperationsTester::getVectorAbsMinTest ) );
      suiteOfTests -> addTest( new TestCallerType( "getVectorLpNormTest", &VectorOperationsTester::getVectorLpNormTest ) );
      suiteOfTests -> addTest( new TestCallerType( "getVectorSumTest", &VectorOperationsTester::getVectorSumTest ) );
      suiteOfTests -> addTest( new TestCallerType( "getVectorDifferenceMaxTest", &VectorOperationsTester::getVectorDifferenceMaxTest ) );
      suiteOfTests -> addTest( new TestCallerType( "getVectorDifferenceMinTest", &VectorOperationsTester::getVectorDifferenceMinTest ) );
      suiteOfTests -> addTest( new TestCallerType( "getVectorDifferenceAbsMaxTest", &VectorOperationsTester::getVectorDifferenceAbsMaxTest ) );
      suiteOfTests -> addTest( new TestCallerType( "getVectorDifferenceAbsMinTest", &VectorOperationsTester::getVectorDifferenceAbsMinTest ) );
      suiteOfTests -> addTest( new TestCallerType( "getVectorDifferenceLpNormTest", &VectorOperationsTester::getVectorDifferenceLpNormTest ) );
      suiteOfTests -> addTest( new TestCallerType( "getVectorDifferenceSumTest", &VectorOperationsTester::getVectorDifferenceSumTest ) );
      suiteOfTests -> addTest( new TestCallerType( "vectorScalarMultiplicationTest", &VectorOperationsTester::vectorScalarMultiplicationTest ) );
      suiteOfTests -> addTest( new TestCallerType( "getSclaraProductTest", &VectorOperationsTester::getVectorScalarProductTest ) );
      suiteOfTests -> addTest( new TestCallerType( "addVectorTest", &VectorOperationsTester::addVectorTest ) );
      /*suiteOfTests -> addTest( new TestCallerType( "alphaXPlusBetaYTest", &VectorOperationsTester::alphaXPlusBetaYTest ) );
      suiteOfTests -> addTest( new TestCallerType( "alphaXPlusBetaZTest", &VectorOperationsTester::alphaXPlusBetaZTest ) );
      suiteOfTests -> addTest( new TestCallerType( "alphaXPlusBetaZPlusYTest", &VectorOperationsTester::alphaXPlusBetaZPlusYTest ) );*/
      suiteOfTests -> addTest( new TestCallerType( "prefixSumTest", &VectorOperationsTester::prefixSumTest ) );
      suiteOfTests -> addTest( new TestCallerType( "exclusivePrefixSumTest", &VectorOperationsTester::exclusivePrefixSumTest ) );
      return suiteOfTests;
   };

   template< typename Vector >
   void setLinearSequence( Vector& deviceVector )
   {
      Vectors::Vector< typename Vector :: RealType, tnlHost > a;
      a. setSize( deviceVector. getSize() );
      for( int i = 0; i < a. getSize(); i ++ )
         a. getData()[ i ] = i;

      ArrayOperations< typename Vector::DeviceType,
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
      Vectors::Vector< typename Vector :: RealType, tnlHost > a;
      a. setSize( deviceVector. getSize() );
      for( int i = 0; i < a. getSize(); i ++ )
         a. getData()[ i ] = 1;

      ArrayOperations< typename Vector::DeviceType,
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
      Vectors::Vector< typename Vector :: RealType, tnlHost > a;
      a. setSize( deviceVector. getSize() );
      for( int i = 0; i < a. getSize(); i ++ )
         a. getData()[ i ] = -i;

      ArrayOperations< typename Vector::DeviceType,
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
      Vectors::Vector< typename Vector::RealType, tnlHost > a;
      a.setSize( deviceVector. getSize() );
      a[ 0 ] = v;
      for( int i = 1; i < a. getSize(); i ++ )
         a.getData()[ i ] = a.getData()[ i-1 ] * -1;

      ArrayOperations< typename Vector::DeviceType,
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
      const int size( 123456 );
      Vectors::Vector< Real, Device > v;
      v. setSize( size );
      setLinearSequence( v );

      CPPUNIT_ASSERT( Vectors::VectorOperations< Device > :: getVectorMax( v ) == size - 1 );
   }

   void getVectorMinTest()
   {
      const int size( 123456 );
      Vectors::Vector< Real, Device > v;
      v. setSize( size );
      setLinearSequence( v );

      CPPUNIT_ASSERT( Vectors::VectorOperations< Device > :: getVectorMin( v ) == 0 );
   }

   void getVectorAbsMaxTest()
   {
      const int size( 123456 );
      Vectors::Vector< Real, Device > v;
      v. setSize( size );
      setNegativeLinearSequence( v );

      CPPUNIT_ASSERT( Vectors::VectorOperations< Device > :: getVectorAbsMax( v ) == size - 1 );
   }

   void getVectorAbsMinTest()
   {
      const int size( 123456 );
      Vectors::Vector< Real, Device > v;
      v. setSize( size );
      setNegativeLinearSequence( v );

      CPPUNIT_ASSERT( Vectors::VectorOperations< Device > :: getVectorAbsMin( v ) == 0 );
   }

   void getVectorLpNormTest()
   {
      const int size( 123456 );
      Vectors::Vector< Real, Device > v;
      v. setSize( size );
      setOnesSequence( v );

      CPPUNIT_ASSERT( Vectors::VectorOperations< Device > :: getVectorLpNorm( v, 2.0 ) == ::sqrt( size ) );
   }

   void getVectorSumTest()
   {
      const int size( 123456 );
      Vectors::Vector< Real, Device > v;
      v. setSize( size );
      setOnesSequence( v );

      CPPUNIT_ASSERT( Vectors::VectorOperations< Device > :: getVectorSum( v ) == size );

      setLinearSequence( v );

      CPPUNIT_ASSERT( Vectors::VectorOperations< Device > :: getVectorSum( v ) == ( ( Real ) size ) * ( ( Real ) size - 1 ) / 2 );
   }

   void getVectorDifferenceMaxTest()
   {
      const int size( 123456 );
      Vectors::Vector< Real, Device > u, v;
      u. setSize( size );
      v. setSize( size );
      setLinearSequence( u );
      setOnesSequence( v );

      CPPUNIT_ASSERT( Vectors::VectorOperations< Device > :: getVectorDifferenceMax( u, v ) == size - 2 );
   }

   void getVectorDifferenceMinTest()
   {
      const int size( 123456 );
      Vectors::Vector< Real, Device > u, v;
      u. setSize( size );
      v. setSize( size );
      setLinearSequence( u );
      setOnesSequence( v );

      CPPUNIT_ASSERT( Vectors::VectorOperations< Device > :: getVectorDifferenceMin( u, v ) == -1 );
      CPPUNIT_ASSERT( Vectors::VectorOperations< Device > :: getVectorDifferenceMin( v, u ) == -123454 );
   }

   void getVectorDifferenceAbsMaxTest()
   {
      const int size( 123456 );
      Vectors::Vector< Real, Device > u, v;
      u. setSize( size );
      v. setSize( size );
      setNegativeLinearSequence( u );
      setOnesSequence( v );

      CPPUNIT_ASSERT( Vectors::VectorOperations< Device > :: getVectorDifferenceAbsMax( u, v ) == size );
   }

   void getVectorDifferenceAbsMinTest()
   {
      const int size( 123456 );
      Vectors::Vector< Real, Device > u, v;
      u. setSize( size );
      v. setSize( size );
      setLinearSequence( u );
      setOnesSequence( v );

      CPPUNIT_ASSERT( Vectors::VectorOperations< Device > :: getVectorDifferenceAbsMin( u, v ) == 0 );
      CPPUNIT_ASSERT( Vectors::VectorOperations< Device > :: getVectorDifferenceAbsMin( v, u ) == 0 );
   }

   void getVectorDifferenceLpNormTest()
   {
      const int size( 1024 );
      Vectors::Vector< Real, Device > u, v;
      u. setSize( size );
      v. setSize( size );
      u. setValue( 3.0 );
      v. setValue( 1.0 );

      CPPUNIT_ASSERT( Vectors::VectorOperations< Device > :: getVectorDifferenceLpNorm( u, v, 1.0 ) == 2.0 * size );
      CPPUNIT_ASSERT( Vectors::VectorOperations< Device > :: getVectorDifferenceLpNorm( u, v, 2.0 ) == ::sqrt( 4.0 * size ) );
   }

   void getVectorDifferenceSumTest()
   {
      const int size( 1024 );
      Vectors::Vector< Real, Device > u, v;
      u. setSize( size );
      v. setSize( size );
      u. setValue( 3.0 );
      v. setValue( 1.0 );

      CPPUNIT_ASSERT( Vectors::VectorOperations< Device > :: getVectorDifferenceSum( u, v ) == 2.0 * size );
   }

   void vectorScalarMultiplicationTest()
   {
      const int size( 1025 );
      Vectors::Vector< Real, Device > u;
      u. setSize( size );
      setLinearSequence( u );

      Vectors::VectorOperations< Device >::vectorScalarMultiplication( u, 3.0 );

      for( int i = 0; i < size; i++ )
         CPPUNIT_ASSERT( u.getElement( i ) == 3.0 * i );
   }

   void getVectorScalarProductTest()
   {
      const int size( 1025 );
      Vectors::Vector< Real, Device > u, v;
      u. setSize( size );
      v. setSize( size );
      setOscilatingSequence( u, 1.0 );
      setOnesSequence( v );

      CPPUNIT_ASSERT( Vectors::VectorOperations< Device > :: getScalarProduct( u, v ) == 1.0 );
   }

   void addVectorTest()
   {
      const int size( 10000 );
      Vectors::Vector< Real, Device > x, y;
      x.setSize( size );
      y.setSize( size );
      setLinearSequence( x );
      setOnesSequence( y );
      Vectors::VectorOperations< Device >::addVector( y, x, 3.0 );

      for( int i = 0; i < size; i ++ )
         CPPUNIT_ASSERT( y.getElement( i ) == 1.0 + 3.0 * i );
   };

   /*void alphaXPlusBetaYTest()
   {
      const int size( 10000 );
      Vector< Real, Device > x, y;
      x.setSize( size );
      y.setSize( size );
      setLinearSequence( x );
      setOnesSequence( y );
      VectorOperations< Device >:: alphaXPlusBetaY( y, x, 3.0, -2.0 );

      for( int i = 0; i < size; i ++ )
         CPPUNIT_ASSERT( y.getElement( i ) == -2.0 + 3.0 * i );
   };

   void alphaXPlusBetaZTest()
   {
      const int size( 10000 );
      Vector< Real, Device > x, y, z;
      x.setSize( size );
      y.setSize( size );
      z.setSize( size );
      setLinearSequence( x );
      setOnesSequence( z );
      VectorOperations< Device >:: alphaXPlusBetaZ( y, x, 3.0, z, -2.0 );

      for( int i = 0; i < size; i ++ )
         CPPUNIT_ASSERT( y.getElement( i ) == -2.0 + 3.0 * i );
   };

   void alphaXPlusBetaZPlusYTest()
   {
      const int size( 10000 );
      Vector< Real, Device > x, y, z;
      x.setSize( size );
      y.setSize( size );
      z.setSize( size );
      setLinearSequence( x );
      setOnesSequence( z );
      setOnesSequence( y );
      VectorOperations< Device >:: alphaXPlusBetaZPlusY( y, x, 3.0, z, -2.0 );

      for( int i = 0; i < size; i ++ )
         CPPUNIT_ASSERT( y.getElement( i ) == -1.0 + 3.0 * i );
   };*/

   void prefixSumTest()
   {
      const int size( 10000 );
      Vectors::Vector< Real, Device > v;
      v.setSize( size );

      setOnesSequence( v );
      v.computePrefixSum();
      for( int i = 0; i < size; i++ )
         CPPUNIT_ASSERT( v.getElement( i ) == i + 1 );

      v.setValue( 0 );
      v.computePrefixSum();
      for( int i = 0; i < size; i++ )
         CPPUNIT_ASSERT( v.getElement( i ) == 0 );

      setLinearSequence( v );
      v.computePrefixSum();
      for( int i = 1; i < size; i++ )
         CPPUNIT_ASSERT( v.getElement( i ) - v.getElement( i - 1 ) == i );

   };

   void exclusivePrefixSumTest()
   {
      const int size( 10000 );
      Vectors::Vector< Real, Device > v;
      v.setSize( size );

      setOnesSequence( v );
      v.computeExclusivePrefixSum();
      for( int i = 0; i < size; i++ )
         CPPUNIT_ASSERT( v.getElement( i ) == i );

      v.setValue( 0 );
      v.computeExclusivePrefixSum();
      for( int i = 0; i < size; i++ )
         CPPUNIT_ASSERT( v.getElement( i ) == 0 );

      setLinearSequence( v );
      v.computeExclusivePrefixSum();
      for( int i = 1; i < size; i++ )
         CPPUNIT_ASSERT( v.getElement( i ) - v.getElement( i - 1 ) == i - 1 );

   };
};

#else
template< typename Real, typename Device >
class VectorOperationsTester
{};
#endif /* HAVE_CPPUNIT */

#endif /* VectorOPERATIONSTESTER_H_ */
