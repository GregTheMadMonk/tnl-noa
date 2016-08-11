/***************************************************************************
                          SparseTester.h  -  description
                             -------------------
    begin                : Jul 11, 2010
    copyright            : (C) 2010 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#ifndef SparseTESTER_H_
#define SparseTESTER_H_

#ifdef HAVE_CPPUNIT
#include <cppunit/TestSuite.h>
#include <cppunit/TestResult.h>
#include <cppunit/TestCaller.h>
#include <cppunit/TestCase.h>
#include <cppunit/Message.h>
#include <TNL/File.h>
#include <TNL/Containers/Vector.h>
#endif 

using namespace TNL;

template< typename Matrix,
          typename TestSetup >
class SparseTesterMatrixSetter
{
   public:

   static bool setup( Matrix& matrix )
   {
      return true;
   }
};


#ifdef HAVE_CUDA
template< typename MatrixType >
__global__ void SparseTester__setElementFastTestCudaKernel( MatrixType* matrix,
                                                                     bool* testResult );
template< typename MatrixType >
__global__ void SparseTester__setElementFast_DiagonalMatrixTestCudaKernel( MatrixType* matrix,
                                                                                    bool* testResult );

template< typename MatrixType >
__global__ void SparseTester__setElementFast_DenseTestCudaKernel1( MatrixType* matrix,
                                                                                  bool* testResult );

template< typename MatrixType >
__global__ void SparseTester__setElementFast_DenseTestCudaKernel2( MatrixType* matrix,
                                                                                  bool* testResult );

template< typename MatrixType >
__global__ void SparseTester__setElementFast_LowerTriangularMatrixTestCudaKernel1( MatrixType* matrix,
                                                                                            bool* testResult );

template< typename MatrixType >
__global__ void SparseTester__setElementFast_LowerTriangularMatrixTestCudaKernel2( MatrixType* matrix,
                                                                                            bool* testResult );

template< typename MatrixType >
__global__ void SparseTester__setRowFast_DiagonalMatrixTestCudaKernel( MatrixType* matrix,
                                                                                bool* testResult );

template< typename MatrixType >
__global__ void SparseTester__setRowFast_DenseTestCudaKernel1( MatrixType* matrix,
                                                                              bool* testResult );

template< typename MatrixType >
__global__ void SparseTester__setRowFast_DenseTestCudaKernel2( MatrixType* matrix,
                                                                              bool* testResult );

template< typename MatrixType >
__global__ void SparseTester__setRowFast_LowerTriangularMatrixTestCudaKernel( MatrixType* matrix,
                                                                                       bool* testResult );

#endif

#ifdef HAVE_CPPUNIT

class SparseTestDefaultSetup
{};

template< typename Matrix,
          typename MatrixSetup = SparseTestDefaultSetup >
class SparseTester : public CppUnit :: TestCase
{
   public:
   typedef Matrix MatrixType;
   typedef typename Matrix::RealType RealType;
   typedef typename Matrix::DeviceType DeviceType;
   typedef typename Matrix::IndexType IndexType;
   typedef Containers::Vector< RealType, DeviceType, IndexType > VectorType;
   typedef Containers::Vector< IndexType, DeviceType, IndexType > IndexVector;
   typedef SparseTester< MatrixType, MatrixSetup > TesterType;
   typedef SparseTesterMatrixSetter< MatrixType, MatrixSetup > MatrixSetter;
   typedef typename CppUnit::TestCaller< TesterType > TestCallerType;

   SparseTester(){};

   virtual
   ~SparseTester(){};

   static CppUnit :: Test* suite()
   {
      String testSuiteName( "SparseTester< " );
      testSuiteName += MatrixType::getType() + " >";

      CppUnit :: TestSuite* suiteOfTests = new CppUnit :: TestSuite( testSuiteName.getString() );
      CppUnit :: TestResult result;

      suiteOfTests->addTest( new TestCallerType( "setDimensionsTest", &TesterType::setDimensionsTest ) );
      suiteOfTests->addTest( new TestCallerType( "setLikeTest", &TesterType::setLikeTest ) );
      suiteOfTests->addTest( new TestCallerType( "setElementTest", &TesterType::setElementTest ) );
      suiteOfTests->addTest( new TestCallerType( "setElementFastTest", &TesterType::setElementFastTest ) );
      suiteOfTests->addTest( new TestCallerType( "setElement_DiagonalMatrixTest", &TesterType::setElement_DiagonalMatrixTest ) );
      suiteOfTests->addTest( new TestCallerType( "setElementFast_DiagonalMatrixTest", &TesterType::setElementFast_DiagonalMatrixTest ) );
      suiteOfTests->addTest( new TestCallerType( "setElement_DenseTest", &TesterType::setElement_DenseTest ) );
      suiteOfTests->addTest( new TestCallerType( "setElementFast_DenseTest", &TesterType::setElementFast_DenseTest ) );
      suiteOfTests->addTest( new TestCallerType( "setElement_LowerTriangularMatrixTest", &TesterType::setElement_LowerTriangularMatrixTest ) );
      suiteOfTests->addTest( new TestCallerType( "setElementFast_LowerTriangularMatrixTest", &TesterType::setElementFast_LowerTriangularMatrixTest ) );
      suiteOfTests->addTest( new TestCallerType( "setRow_DiagonalMatrixTest", &TesterType::setRow_DiagonalMatrixTest ) );
      suiteOfTests->addTest( new TestCallerType( "setRowFast_DiagonalMatrixTest", &TesterType::setRowFast_DiagonalMatrixTest ) );
      suiteOfTests->addTest( new TestCallerType( "setRow_DenseTest", &TesterType::setRow_DenseTest ) );
      suiteOfTests->addTest( new TestCallerType( "setRowFast_DenseTest", &TesterType::setRowFast_DenseTest ) );
      suiteOfTests->addTest( new TestCallerType( "setRow_LowerTriangularMatrixTest", &TesterType::setRow_LowerTriangularMatrixTest ) );
      suiteOfTests->addTest( new TestCallerType( "setRowFast_LowerTriangularMatrixTest", &TesterType::setRowFast_LowerTriangularMatrixTest ) );
      suiteOfTests->addTest( new TestCallerType( "addElementTest", &TesterType::addElementTest ) );
      suiteOfTests->addTest( new TestCallerType( "vectorProduct_DiagonalMatrixTest", &TesterType::vectorProduct_DiagonalMatrixTest ) );
      suiteOfTests->addTest( new TestCallerType( "vectorProduct_DenseTest", &TesterType::vectorProduct_DenseTest ) );
      suiteOfTests->addTest( new TestCallerType( "vectorProduct_LowerTriangularMatrixTest", &TesterType::vectorProduct_LowerTriangularMatrixTest ) );
      /*suiteOfTests -> addTest( new TestCallerType( "matrixTranspositionTest", &TesterType::matrixTranspositionTest ) );
      suiteOfTests -> addTest( new TestCallerType( "addMatrixTest", &TesterType::addMatrixTest ) );*/

      return suiteOfTests;
   }

   void setDimensionsTest()
   {
      MatrixType m;
      MatrixSetter::setup( m );
      m.setDimensions( 10, 10 );
      CPPUNIT_ASSERT( m.getRows() == 10 );
      CPPUNIT_ASSERT( m.getColumns() == 10 );
   }

   void setLikeTest()
   {
      MatrixType m1, m2;
      MatrixSetter::setup( m1 );
      MatrixSetter::setup( m2 );
      m1.setDimensions( 10, 10 );
      IndexVector rowLengths;
      rowLengths.setSize( m1.getRows() );
      rowLengths.setValue( 5 );
      m1.setCompressedRowsLengths( rowLengths );
      m2.setLike( m1 );
      CPPUNIT_ASSERT( m1.getRows() == m2.getRows() );
   }

   /****
    * Set element tests
    */
   void setElementTest()
   {
      MatrixType m;
      MatrixSetter::setup( m );
      m.setDimensions( 10, 10 );
      IndexVector rowLengths;
      rowLengths.setSize( m.getRows() );
      rowLengths.setValue( 7 );
      m.setCompressedRowsLengths( rowLengths );

      for( int i = 0; i < 7; i++ )
         CPPUNIT_ASSERT( m.setElement( 0, i, i ) );

      //CPPUNIT_ASSERT( m.setElement( 0, 8, 8 ) == false );

      for( int i = 0; i < 7; i++ )
         CPPUNIT_ASSERT( m.getElement( 0, i ) == i );
   }


   void setElementFastTest()
   {
      MatrixType m;
      MatrixSetter::setup( m );
      m.setDimensions( 10, 10 );
      IndexVector rowLengths;
      rowLengths.setSize( m.getRows() );
      rowLengths.setValue( 7 );
      m.setCompressedRowsLengths( rowLengths );

      if( std::is_same< DeviceType, Devices::Host >::value )
      {
         for( int i = 0; i < 7; i++ )
            CPPUNIT_ASSERT( m.setElementFast( 0, i, i ) );
         //CPPUNIT_ASSERT( m.setElementFast( 0, 8, 8 ) == false );
      }

      if( std::is_same< DeviceType, Devices::Cuda >::value )
      {
#ifdef HAVE_CUDA
         MatrixType* kernel_matrix = Devices::Cuda::passToDevice( m );
         bool testResult( true );
         bool* kernel_testResult = Devices::Cuda::passToDevice( testResult );
         checkCudaDevice;
         dim3 cudaBlockSize( 256 ), cudaGridSize( 1 );
         SparseTester__setElementFastTestCudaKernel< MatrixType >
                                                            <<< cudaGridSize, cudaBlockSize >>>
                                                            ( kernel_matrix,
                                                              kernel_testResult );
         CPPUNIT_ASSERT( Devices::Cuda::passFromDevice( kernel_testResult ) );
         Devices::Cuda::freeFromDevice( kernel_matrix );
         Devices::Cuda::freeFromDevice( kernel_testResult );
         checkCudaDevice;
#endif
      }

      for( int i = 0; i < 7; i++ )
         CPPUNIT_ASSERT( m.getElement( 0, i ) == i );
   }

   void setElement_DiagonalMatrixTest()
   {
      MatrixType m;
      MatrixSetter::setup( m );
      m.setDimensions( 10, 10 );
      IndexVector rowLengths;
      rowLengths.setSize( m.getRows() );
      rowLengths.setValue( 7 );
      m.setCompressedRowsLengths( rowLengths );

      for( int i = 0; i < 10; i++ )
         m.setElement( i, i, i );

      for( int i = 0; i < 10; i++ )
      {
         for( int j = 0; j < 10; j++ )
         {
            if( i == j )
               CPPUNIT_ASSERT( m.getElement( i, j ) == i );
            else
               CPPUNIT_ASSERT( m.getElement( i, j ) == 0 );
         }
      }
   }

   void setElementFast_DiagonalMatrixTest()
   {
      MatrixType m;
      MatrixSetter::setup( m );
      m.setDimensions( 10, 10 );
      IndexVector rowLengths;
      rowLengths.setSize( m.getRows() );
      rowLengths.setValue( 7 );
      m.setCompressedRowsLengths( rowLengths );

      if( std::is_same< DeviceType, Devices::Host >::value )
      {
         for( int i = 0; i < 10; i++ )
            m.setElementFast( i, i, i );
      }
      if( std::is_same< DeviceType, Devices::Cuda >::value )
      {
#ifdef HAVE_CUDA
         MatrixType* kernel_matrix = Devices::Cuda::passToDevice( m );
         bool testResult( true );
         bool* kernel_testResult = Devices::Cuda::passToDevice( testResult );
         checkCudaDevice;
         dim3 cudaBlockSize( 256 ), cudaGridSize( 1 );
         SparseTester__setElementFast_DiagonalMatrixTestCudaKernel< MatrixType >
                                                                           <<< cudaGridSize, cudaBlockSize >>>
                                                                           ( kernel_matrix,
                                                                             kernel_testResult );
         CPPUNIT_ASSERT( Devices::Cuda::passFromDevice( kernel_testResult ) );
         Devices::Cuda::freeFromDevice( kernel_matrix );
         Devices::Cuda::freeFromDevice( kernel_testResult );
         checkCudaDevice;
#endif
      }

      for( int i = 0; i < 10; i++ )
      {
         for( int j = 0; j < 10; j++ )
         {
            if( i == j )
               CPPUNIT_ASSERT( m.getElement( i, j ) == i );
            else
               CPPUNIT_ASSERT( m.getElement( i, j ) == 0 );
         }
      }
   }

   void setElement_DenseTest()
   {
      MatrixType m;
      MatrixSetter::setup( m );
      m.setDimensions( 10, 10 );
      IndexVector rowLengths;
      rowLengths.setSize( m.getRows() );
      rowLengths.setValue( 10 );
      m.setCompressedRowsLengths( rowLengths );

      for( int i = 0; i < 10; i++ )
         m.setElement( i, i, i );
      for( int i = 0; i < 10; i++ )
         for( int j = 0; j < 10; j++ )
            m.addElement( i, j, 1, 0.5 );

      for( int i = 0; i < 10; i++ )
         for( int j = 0; j < 10; j++ )
            if( i == j )
               CPPUNIT_ASSERT( m.getElement( i, j ) == 1.0+0.5*i );
            else
               CPPUNIT_ASSERT( m.getElement( i, j ) == 1.0 );

      m.reset();
      m.setDimensions( 10, 10 );
      m.setCompressedRowsLengths( rowLengths );
      for( int i = 9; i >= 0; i-- )
         for( int j = 9; j >= 0; j-- )
            m.setElement( i, j, i+j );

      for( int i = 9; i >= 0; i-- )
         for( int j = 9; j >= 0; j-- )
            CPPUNIT_ASSERT( m.getElement( i, j ) == i+j );
   }

   void setElementFast_DenseTest()
   {
      MatrixType m;
      MatrixSetter::setup( m );
      m.setDimensions( 10, 10 );
      IndexVector rowLengths;
      rowLengths.setSize( m.getRows() );
      rowLengths.setValue( 10 );
      m.setCompressedRowsLengths( rowLengths );

      if( std::is_same< DeviceType, Devices::Host >::value )
      {
         for( int i = 0; i < 10; i++ )
            m.setElementFast( i, i, i );
         for( int i = 0; i < 10; i++ )
            for( int j = 0; j < 10; j++ )
               m.addElementFast( i, j, 1, 0.5 );
      }
      if( std::is_same< DeviceType, Devices::Cuda >::value )
      {
#ifdef HAVE_CUDA
         MatrixType* kernel_matrix = Devices::Cuda::passToDevice( m );
         bool testResult( true );
         bool* kernel_testResult = Devices::Cuda::passToDevice( testResult );
         checkCudaDevice;
         dim3 cudaBlockSize( 256 ), cudaGridSize( 1 );
         SparseTester__setElementFast_DenseTestCudaKernel1< MatrixType >
                                                                         <<< cudaGridSize, cudaBlockSize >>>
                                                                         ( kernel_matrix,
                                                                           kernel_testResult );
         CPPUNIT_ASSERT( Devices::Cuda::passFromDevice( kernel_testResult ) );
         Devices::Cuda::freeFromDevice( kernel_matrix );
         Devices::Cuda::freeFromDevice( kernel_testResult );
         checkCudaDevice;
#endif
      }

      for( int i = 0; i < 10; i++ )
         for( int j = 0; j < 10; j++ )
            if( i == j )
               CPPUNIT_ASSERT( m.getElement( i, j ) == 1.0+0.5*i );
            else
               CPPUNIT_ASSERT( m.getElement( i, j ) == 1.0 );

      m.reset();
      m.setDimensions( 10, 10 );
      m.setCompressedRowsLengths( rowLengths );
      if( std::is_same< DeviceType, Devices::Host >::value )
      {
         for( int i = 9; i >= 0; i-- )
            for( int j = 9; j >= 0; j-- )
               m.setElementFast( i, j, i+j );
      }
      if( std::is_same< DeviceType, Devices::Cuda >::value )
      {
#ifdef HAVE_CUDA
         MatrixType* kernel_matrix = Devices::Cuda::passToDevice( m );
         bool testResult( true );
         bool* kernel_testResult = Devices::Cuda::passToDevice( testResult );
         checkCudaDevice;
         dim3 cudaBlockSize( 256 ), cudaGridSize( 1 );
         SparseTester__setElementFast_DenseTestCudaKernel2< MatrixType >
                                                                         <<< cudaGridSize, cudaBlockSize >>>
                                                                         ( kernel_matrix,
                                                                           kernel_testResult );
         CPPUNIT_ASSERT( Devices::Cuda::passFromDevice( kernel_testResult ) );
         Devices::Cuda::freeFromDevice( kernel_matrix );
         Devices::Cuda::freeFromDevice( kernel_testResult );
         checkCudaDevice;
#endif
      }

      for( int i = 9; i >= 0; i-- )
         for( int j = 9; j >= 0; j-- )
            CPPUNIT_ASSERT( m.getElement( i, j ) == i+j );
   }


   void setElement_LowerTriangularMatrixTest()
   {
      MatrixType m;
      MatrixSetter::setup( m );
      m.setDimensions( 10, 10 );
      IndexVector rowLengths;
      rowLengths.setSize( m.getRows() );
      for( int i = 0; i < 10; i++ )
         rowLengths.setElement( i, i+1 );
      m.setCompressedRowsLengths( rowLengths );

      for( int i = 0; i < 10; i++ )
         for( int j = 0; j <= i; j++ )
            m.setElement( i, j, i + j );

      for( int i = 0; i < 10; i++ )
         for( int j = 0; j < 10; j++ )
            if( j <= i )
               CPPUNIT_ASSERT( m.getElement( i, j ) == i + j );
            else
               CPPUNIT_ASSERT( m.getElement( i, j ) == 0 );

      m.reset();
      m.setDimensions( 10, 10 );
      m.setCompressedRowsLengths( rowLengths );
      for( int i = 9; i >= 0; i-- )
         for( int j = i; j >= 0; j-- )
            m.setElement( i, j, i + j );

      for( int i = 0; i < 10; i++ )
         for( int j = 0; j < 10; j++ )
            if( j <= i )
               CPPUNIT_ASSERT( m.getElement( i, j ) == i + j );
            else
               CPPUNIT_ASSERT( m.getElement( i, j ) == 0 );
   }

   void setElementFast_LowerTriangularMatrixTest()
   {
      MatrixType m;
      MatrixSetter::setup( m );
      m.setDimensions( 10, 10 );
      IndexVector rowLengths;
      rowLengths.setSize( m.getRows() );
      for( int i = 0; i < 10; i++ )
         rowLengths.setElement( i, i+1 );
      m.setCompressedRowsLengths( rowLengths );

      if( std::is_same< DeviceType, Devices::Host >::value )
      {
         for( int i = 0; i < 10; i++ )
            for( int j = 0; j <= i; j++ )
               m.setElementFast( i, j, i + j );
      }
      if( std::is_same< DeviceType, Devices::Cuda >::value )
      {
#ifdef HAVE_CUDA
         MatrixType* kernel_matrix = Devices::Cuda::passToDevice( m );
         bool testResult( true );
         bool* kernel_testResult = Devices::Cuda::passToDevice( testResult );
         checkCudaDevice;
         dim3 cudaBlockSize( 256 ), cudaGridSize( 1 );
         SparseTester__setElementFast_LowerTriangularMatrixTestCudaKernel1< MatrixType >
                                                                                   <<< cudaGridSize, cudaBlockSize >>>
                                                                                   ( kernel_matrix,
                                                                                     kernel_testResult );
         CPPUNIT_ASSERT( Devices::Cuda::passFromDevice( kernel_testResult ) );
         Devices::Cuda::freeFromDevice( kernel_matrix );
         Devices::Cuda::freeFromDevice( kernel_testResult );
         checkCudaDevice;
#endif
      }

      for( int i = 0; i < 10; i++ )
         for( int j = 0; j < 10; j++ )
            if( j <= i )
               CPPUNIT_ASSERT( m.getElement( i, j ) == i + j );
            else
               CPPUNIT_ASSERT( m.getElement( i, j ) == 0 );

      m.reset();
      m.setDimensions( 10, 10 );
      m.setCompressedRowsLengths( rowLengths );
      if( std::is_same< DeviceType, Devices::Host >::value )
      {
         for( int i = 9; i >= 0; i-- )
            for( int j = i; j >= 0; j-- )
               m.setElementFast( i, j, i + j );
      }
      if( std::is_same< DeviceType, Devices::Cuda >::value )
      {
#ifdef HAVE_CUDA
         MatrixType* kernel_matrix = Devices::Cuda::passToDevice( m );
         bool testResult( true );
         bool* kernel_testResult = Devices::Cuda::passToDevice( testResult );
         checkCudaDevice;
         dim3 cudaBlockSize( 256 ), cudaGridSize( 1 );
         SparseTester__setElementFast_LowerTriangularMatrixTestCudaKernel2< MatrixType >
                                                                                   <<< cudaGridSize, cudaBlockSize >>>
                                                                                   ( kernel_matrix,
                                                                                     kernel_testResult );
         CPPUNIT_ASSERT( Devices::Cuda::passFromDevice( kernel_testResult ) );
         Devices::Cuda::freeFromDevice( kernel_matrix );
         Devices::Cuda::freeFromDevice( kernel_testResult );
         checkCudaDevice;
#endif
      }

      for( int i = 0; i < 10; i++ )
         for( int j = 0; j < 10; j++ )
            if( j <= i )
               CPPUNIT_ASSERT( m.getElement( i, j ) == i + j );
            else
               CPPUNIT_ASSERT( m.getElement( i, j ) == 0 );
   }


   void addElementTest()
   {
      MatrixType m;
      MatrixSetter::setup( m );
      m.setDimensions( 10, 10 );
      IndexVector rowLengths;
      rowLengths.setSize( m.getRows() );
      rowLengths.setValue( 7 );
      m.setCompressedRowsLengths( rowLengths );
      for( int i = 0; i < 10; i++ )
         m.setElement( i, i, i );
      for( int i = 0; i < 10; i++ )
         for( int j = 0; j < 10; j++ )
            if( std::abs( i - j ) <= 1 )
               m.addElement( i, j, 1 );

      for( int i = 0; i < 10; i++ )
         for( int j = 0; j < 10; j++ )
            if( i == j )
               CPPUNIT_ASSERT( m.getElement( i, i ) == i + 1 );
            else
               if( std::abs( i - j ) == 1 )
                  CPPUNIT_ASSERT( m.getElement( i, j ) == 1 );
               else
                  CPPUNIT_ASSERT( m.getElement( i, j ) == 0 );
   }

   /****
    * Set row tests
    */
   void setRow_DiagonalMatrixTest()
   {
      MatrixType m;
      MatrixSetter::setup( m );
      m.setDimensions( 10, 10 );
      IndexVector rowLengths;
      rowLengths.setSize( m.getRows() );
      rowLengths.setValue( 7 );
      m.setCompressedRowsLengths( rowLengths );
      RealType values[ 1 ];
      IndexType columnIndexes[ 1 ];

      for( int i = 0; i < 10; i++ )
      {
         values[ 0 ] = i;
         columnIndexes[ 0 ] = i;
         m.setRow( i, columnIndexes, values, 1 );
      }

      for( int i = 0; i < 10; i++ )
      {
         for( int j = 0; j < 10; j++ )
         {
            if( i == j )
               CPPUNIT_ASSERT( m.getElement( i, j ) == i );
            else
               CPPUNIT_ASSERT( m.getElement( i, j ) == 0 );
         }
      }
   }

   void setRowFast_DiagonalMatrixTest()
   {
      MatrixType m;
      MatrixSetter::setup( m );
      m.setDimensions( 10, 10 );
      IndexVector rowLengths;
      rowLengths.setSize( m.getRows() );
      rowLengths.setValue( 7 );
      m.setCompressedRowsLengths( rowLengths );


      if( std::is_same< DeviceType, Devices::Host >::value )
      {
         RealType values[ 1 ];
         IndexType columnIndexes[ 1 ];
         for( int i = 0; i < 10; i++ )
         {
            values[ 0 ] = i;
            columnIndexes[ 0 ] = i;
            m.setRowFast( i, columnIndexes, values, 1 );
         }
      }
      if( std::is_same< DeviceType, Devices::Cuda >::value )
      {
#ifdef HAVE_CUDA
         MatrixType* kernel_matrix = Devices::Cuda::passToDevice( m );
         bool testResult( true );
         bool* kernel_testResult = Devices::Cuda::passToDevice( testResult );
         checkCudaDevice;
         dim3 cudaBlockSize( 256 ), cudaGridSize( 1 );
         int sharedMemory = 100 * ( sizeof( IndexType ) + sizeof( RealType ) );
         SparseTester__setRowFast_DiagonalMatrixTestCudaKernel< MatrixType >
                                                                       <<< cudaGridSize, cudaBlockSize, sharedMemory >>>
                                                                       ( kernel_matrix,
                                                                         kernel_testResult );
         CPPUNIT_ASSERT( Devices::Cuda::passFromDevice( kernel_testResult ) );
         Devices::Cuda::freeFromDevice( kernel_matrix );
         Devices::Cuda::freeFromDevice( kernel_testResult );
         checkCudaDevice;
#endif
      }

      for( int i = 0; i < 10; i++ )
      {
         for( int j = 0; j < 10; j++ )
         {
            if( i == j )
               CPPUNIT_ASSERT( m.getElement( i, j ) == i );
            else
               CPPUNIT_ASSERT( m.getElement( i, j ) == 0 );
         }
      }
   }

   void setRow_DenseTest()
   {
      MatrixType m;
      MatrixSetter::setup( m );
      m.setDimensions( 10, 10 );
      IndexVector rowLengths;
      rowLengths.setSize( m.getRows() );
      rowLengths.setValue( 10 );
      m.setCompressedRowsLengths( rowLengths );
      RealType values[ 10 ];
      IndexType columnIndexes[ 10 ];

      for( int i = 0; i < 10; i++ )
         columnIndexes[ i ] = i;
      for( int i = 0; i < 10; i++ )
      {
         for( int j = 0; j < 10; j++ )
            if( i == j )
               values[ i ] = 1.0 + 0.5 * j;
            else
               values[ j ] = 1.0;

         m.setRow( i, columnIndexes, values, 10 );
      }

      for( int i = 0; i < 10; i++ )
         for( int j = 0; j < 10; j++ )
            if( i == j )
               CPPUNIT_ASSERT( m.getElement( i, j ) == 1.0+0.5*i );
            else
               CPPUNIT_ASSERT( m.getElement( i, j ) == 1.0 );

      m.reset();
      m.setDimensions( 10, 10 );
      m.setCompressedRowsLengths( rowLengths );
      for( int i = 9; i >= 0; i-- )
      {
         for( int j = 9; j >= 0; j-- )
            values[ j ] = i+j;
         m.setRow( i, columnIndexes, values, 10 );
      }

      for( int i = 9; i >= 0; i-- )
         for( int j = 9; j >= 0; j-- )
            CPPUNIT_ASSERT( m.getElement( i, j ) == i+j );
   }

   void setRowFast_DenseTest()
   {
      MatrixType m;
      MatrixSetter::setup( m );
      m.setDimensions( 10, 10 );
      IndexVector rowLengths;
      rowLengths.setSize( m.getRows() );
      rowLengths.setValue( 10 );
      m.setCompressedRowsLengths( rowLengths );

      RealType values[ 10 ];
      IndexType columnIndexes[ 10 ];
      for( int i = 0; i < 10; i++ )
         columnIndexes[ i ] = i;

      if( std::is_same< DeviceType, Devices::Host >::value )
      {
         for( int i = 0; i < 10; i++ )
         {
            for( int j = 0; j < 10; j++ )
               if( i == j )
                  values[ i ] = 1.0 + 0.5 * j;
               else
                  values[ j ] = 1.0;

            m.setRowFast( i, columnIndexes, values, 10 );
         }
      }
      if( std::is_same< DeviceType, Devices::Cuda >::value )
      {
#ifdef HAVE_CUDA
         MatrixType* kernel_matrix = Devices::Cuda::passToDevice( m );
         bool testResult( true );
         bool* kernel_testResult = Devices::Cuda::passToDevice( testResult );
         checkCudaDevice;
         dim3 cudaBlockSize( 256 ), cudaGridSize( 1 );
         int sharedMemory = 100 * ( sizeof( IndexType ) + sizeof( RealType ) );
         SparseTester__setRowFast_DenseTestCudaKernel1< MatrixType >
                                                                        <<< cudaGridSize, cudaBlockSize, sharedMemory >>>
                                                                        ( kernel_matrix,
                                                                          kernel_testResult );
         CPPUNIT_ASSERT( Devices::Cuda::passFromDevice( kernel_testResult ) );
         Devices::Cuda::freeFromDevice( kernel_matrix );
         Devices::Cuda::freeFromDevice( kernel_testResult );
         checkCudaDevice;
#endif
      }

      for( int i = 0; i < 10; i++ )
         for( int j = 0; j < 10; j++ )
            if( i == j )
               CPPUNIT_ASSERT( m.getElement( i, j ) == 1.0+0.5*i );
            else
               CPPUNIT_ASSERT( m.getElement( i, j ) == 1.0 );

      m.reset();
      m.setDimensions( 10, 10 );
      m.setCompressedRowsLengths( rowLengths );

      if( std::is_same< DeviceType, Devices::Host >::value )
      {
         for( int i = 9; i >= 0; i-- )
         {
            for( int j = 9; j >= 0; j-- )
               values[ j ] = i+j;
            m.setRowFast( i, columnIndexes, values, 10 );
         }
      }
      if( std::is_same< DeviceType, Devices::Cuda >::value )
      {
#ifdef HAVE_CUDA
         MatrixType* kernel_matrix = Devices::Cuda::passToDevice( m );
         bool testResult( true );
         bool* kernel_testResult = Devices::Cuda::passToDevice( testResult );
         checkCudaDevice;
         dim3 cudaBlockSize( 256 ), cudaGridSize( 1 );
         int sharedMemory = 100 * ( sizeof( IndexType ) + sizeof( RealType ) );
         SparseTester__setRowFast_DenseTestCudaKernel2< MatrixType >
                                                                     <<< cudaGridSize, cudaBlockSize, sharedMemory >>>
                                                                     ( kernel_matrix,
                                                                       kernel_testResult );
         CPPUNIT_ASSERT( Devices::Cuda::passFromDevice( kernel_testResult ) );
         Devices::Cuda::freeFromDevice( kernel_matrix );
         Devices::Cuda::freeFromDevice( kernel_testResult );
         checkCudaDevice;
#endif
      }

      for( int i = 9; i >= 0; i-- )
         for( int j = 9; j >= 0; j-- )
            CPPUNIT_ASSERT( m.getElement( i, j ) == i+j );
   }

   void setRow_LowerTriangularMatrixTest()
   {
      MatrixType m;
      MatrixSetter::setup( m );
      m.setDimensions( 10, 10 );
      IndexVector rowLengths;
      rowLengths.setSize( m.getRows() );
      for( int i = 0; i < 10; i++ )
         rowLengths.setElement( i, i+1 );
      m.setCompressedRowsLengths( rowLengths );


      RealType values[ 10 ];
      IndexType columnIndexes[ 10 ];

      for( int i = 0; i < 10; i++ )
         columnIndexes[ i ] = i;

      for( int i = 0; i < 10; i++ )
      {
         for( int j = 0; j <= i; j++ )
            values[ j ] = i + j;
         m.setRow( i, columnIndexes, values, i + 1 );
      }

      for( int i = 0; i < 10; i++ )
         for( int j = 0; j < 10; j++ )
            if( j <= i )
               CPPUNIT_ASSERT( m.getElement( i, j ) == i + j );
            else
               CPPUNIT_ASSERT( m.getElement( i, j ) == 0 );

      m.reset();
      m.setDimensions( 10, 10 );
      m.setCompressedRowsLengths( rowLengths );
      for( int i = 9; i >= 0; i-- )
      {
         for( int j = i; j >= 0; j-- )
            values[ j ] = i + j;
         m.setRow( i, columnIndexes, values, i + 1 );
      }

      for( int i = 0; i < 10; i++ )
         for( int j = 0; j < 10; j++ )
            if( j <= i )
               CPPUNIT_ASSERT( m.getElement( i, j ) == i + j );
            else
               CPPUNIT_ASSERT( m.getElement( i, j ) == 0 );
   }

   void setRowFast_LowerTriangularMatrixTest()
   {
      MatrixType m;
      MatrixSetter::setup( m );
      m.setDimensions( 10, 10 );
      IndexVector rowLengths;
      rowLengths.setSize( m.getRows() );
      for( int i = 0; i < 10; i++ )
         rowLengths.setElement( i, i+1 );
      m.setCompressedRowsLengths( rowLengths );


      RealType values[ 10 ];
      IndexType columnIndexes[ 10 ];
      for( int i = 0; i < 10; i++ )
         columnIndexes[ i ] = i;

      if( std::is_same< DeviceType, Devices::Host >::value )
      {
         for( int i = 0; i < 10; i++ )
         {
            for( int j = 0; j <= i; j++ )
               values[ j ] = i + j;
            m.setRowFast( i, columnIndexes, values, i + 1 );
         }
      }
      if( std::is_same< DeviceType, Devices::Cuda >::value )
      {
#ifdef HAVE_CUDA
         MatrixType* kernel_matrix = Devices::Cuda::passToDevice( m );
         bool testResult( true );
         bool* kernel_testResult = Devices::Cuda::passToDevice( testResult );
         checkCudaDevice;
         dim3 cudaBlockSize( 256 ), cudaGridSize( 1 );
         int sharedMemory = 100 * ( sizeof( IndexType ) + sizeof( RealType ) );
         SparseTester__setRowFast_LowerTriangularMatrixTestCudaKernel< MatrixType >
                                                                              <<< cudaGridSize, cudaBlockSize, sharedMemory >>>
                                                                              ( kernel_matrix,
                                                                                kernel_testResult );
         CPPUNIT_ASSERT( Devices::Cuda::passFromDevice( kernel_testResult ) );
         Devices::Cuda::freeFromDevice( kernel_matrix );
         Devices::Cuda::freeFromDevice( kernel_testResult );
         checkCudaDevice;
#endif
      }

      for( int i = 0; i < 10; i++ )
         for( int j = 0; j < 10; j++ )
            if( j <= i )
               CPPUNIT_ASSERT( m.getElement( i, j ) == i + j );
            else
               CPPUNIT_ASSERT( m.getElement( i, j ) == 0 );

      m.reset();
      m.setDimensions( 10, 10 );
      m.setCompressedRowsLengths( rowLengths );

      if( std::is_same< DeviceType, Devices::Host >::value )
      {
         for( int i = 9; i >= 0; i-- )
         {
            for( int j = i; j >= 0; j-- )
               values[ j ] = i + j;
            m.setRowFast( i, columnIndexes, values, i + 1 );
         }
      }
      if( std::is_same< DeviceType, Devices::Cuda >::value )
      {
#ifdef HAVE_CUDA
         MatrixType* kernel_matrix = Devices::Cuda::passToDevice( m );
         bool testResult( true );
         bool* kernel_testResult = Devices::Cuda::passToDevice( testResult );
         checkCudaDevice;
         dim3 cudaBlockSize( 256 ), cudaGridSize( 1 );
         int sharedMemory = 100 * ( sizeof( IndexType ) + sizeof( RealType ) );
         SparseTester__setRowFast_LowerTriangularMatrixTestCudaKernel< MatrixType >
                                                                              <<< cudaGridSize, cudaBlockSize, sharedMemory >>>
                                                                              ( kernel_matrix,
                                                                                kernel_testResult );
         CPPUNIT_ASSERT( Devices::Cuda::passFromDevice( kernel_testResult ) );
         Devices::Cuda::freeFromDevice( kernel_matrix );
         Devices::Cuda::freeFromDevice( kernel_testResult );
         checkCudaDevice;
#endif
      }

      for( int i = 0; i < 10; i++ )
         for( int j = 0; j < 10; j++ )
            if( j <= i )
               CPPUNIT_ASSERT( m.getElement( i, j ) == i + j );
            else
               CPPUNIT_ASSERT( m.getElement( i, j ) == 0 );
   }


   void vectorProduct_DiagonalMatrixTest()
   {
      const int size = 10;
      VectorType v, w;
      v.setSize( size );
      w.setSize( size );
      MatrixType m;
      MatrixSetter::setup( m );
      m.setDimensions( size, size );
      IndexVector rowLengths;
      rowLengths.setSize( m.getRows() );
      rowLengths.setValue( 7 );
      m.setCompressedRowsLengths( rowLengths );
      for( int i = 0; i < size; i++ )
      {
         v.setElement( i, i );
         m.setElement( i, i, i );
      }
      m.vectorProduct( v, w );

      for( int i = 0; i < size; i++ )
         CPPUNIT_ASSERT( w.getElement( i ) == i*i );
   }

   void vectorProduct_DenseTest()
   {
      const int size = 10;
      VectorType v, w;
      v.setSize( size );
      w.setSize( size );
      MatrixType m;
      MatrixSetter::setup( m );
      m.setDimensions( size, size );
      IndexVector rowLengths;
      rowLengths.setSize( m.getRows() );
      rowLengths.setValue( size );
      m.setCompressedRowsLengths( rowLengths );
      for( int i = 0; i < size; i++ )
      {
         for( int j = 0; j < size; j++ )
            m.setElement( i, j, i );
         v.setElement( i, 1 );
      }
      m.vectorProduct( v, w );

      for( int i = 0; i < size; i++ )
         CPPUNIT_ASSERT( w.getElement( i ) == i*size );
   }

   void vectorProduct_LowerTriangularMatrixTest()
   {
      const int size = 10;
      VectorType v, w;
      v.setSize( size );
      w.setSize( size );
      MatrixType m;
      MatrixSetter::setup( m );
      m.setDimensions( size, size );
      IndexVector rowLengths;
      rowLengths.setSize( m.getRows() );
      rowLengths.setValue( size );
      m.setCompressedRowsLengths( rowLengths );
      for( int i = 0; i < size; i++ )
      {
         for( int j = 0; j <= i; j++ )
            m.setElement( i, j, i );
         v.setElement( i, 1 );
      }
      m.vectorProduct( v, w );

      for( int i = 0; i < size; i++ )
         CPPUNIT_ASSERT( w.getElement( i ) == i*( i + 1 ) );
   }


   void addMatrixTest()
   {
   }

   void matrixTranspositionTest()
   {
   }
};

#ifdef HAVE_CUDA
   template< typename MatrixType >
   __global__ void SparseTester__setElementFastTestCudaKernel( MatrixType* matrix,
                                                                        bool* testResult )
   {
      if( threadIdx.x == 0 )
      {
         for( int i = 0; i < 7; i++ )
            if( matrix->setElementFast( 0, i, i ) != true )
               testResult = false;
         if( matrix->setElementFast( 0, 8, 8 ) == true )
            testResult = false;
      }
   }

   template< typename MatrixType >
   __global__ void SparseTester__setElementFast_DiagonalMatrixTestCudaKernel( MatrixType* matrix,
                                                                                       bool* testResult )
   {
      if( threadIdx.x < matrix->getRows() )
         matrix->setElementFast( threadIdx.x, threadIdx.x, threadIdx.x );
   }

   template< typename MatrixType >
   __global__ void SparseTester__setElementFast_DenseTestCudaKernel1( MatrixType* matrix,
                                                                                     bool* testResult )
   {
      const typename MatrixType::IndexType i = threadIdx.x;
      if( i < matrix->getRows() )
      {
         matrix->setElementFast( i, i, i );
         for( int j = 0; j < matrix->getColumns(); j++ )
            matrix->addElementFast( i, j, 1, 0.5 );
      }
   }

   template< typename MatrixType >
   __global__ void SparseTester__setElementFast_DenseTestCudaKernel2( MatrixType* matrix,
                                                                                     bool* testResult )
   {
      const typename MatrixType::IndexType i = threadIdx.x;
      if( i < matrix->getRows() )
      {
         for( int j = matrix->getColumns() -1; j >= 0; j-- )
            matrix->setElementFast( i, j, i + j );
      }
   }

   template< typename MatrixType >
   __global__ void SparseTester__setElementFast_LowerTriangularMatrixTestCudaKernel1( MatrixType* matrix,
                                                                                               bool* testResult )
   {
      const typename MatrixType::IndexType i = threadIdx.x;
      if( i < matrix->getRows() )
      {
         for( int j = 0; j <= i; j++ )
            matrix->setElementFast( i, j, i + j );
      }
   }

   template< typename MatrixType >
   __global__ void SparseTester__setElementFast_LowerTriangularMatrixTestCudaKernel2( MatrixType* matrix,
                                                                                               bool* testResult )
   {
      const typename MatrixType::IndexType i = threadIdx.x;
      if( i < matrix->getRows() )
      {
         for( int j = i; j >= 0; j-- )
            matrix->setElementFast( i, j, i + j );
      }
   }

   /****
    * Set row tests kernels
    */
   template< typename MatrixType >
   __global__ void SparseTester__setRowFast_DiagonalMatrixTestCudaKernel( MatrixType* matrix,
                                                                                   bool* testResult )
   {
      typedef typename MatrixType::RealType RealType;
      typedef typename MatrixType::IndexType IndexType;

      const IndexType row = threadIdx.x;
      if( row >= matrix->getRows() )
         return;

      IndexType* columnIndexes = getSharedMemory< IndexType >();
      RealType* valuesBase = ( RealType* ) & columnIndexes[ matrix->getColumns() ];
      RealType* values = &valuesBase[ row ];

      columnIndexes[ row ] = row;
      values[ 0 ] = row;

      matrix->setRowFast( row, &columnIndexes[ row ], values, 1 );
   }

   template< typename MatrixType >
   __global__ void SparseTester__setRowFast_DenseTestCudaKernel1( MatrixType* matrix,
                                                                                 bool* testResult )
   {
      typedef typename MatrixType::RealType RealType;
      typedef typename MatrixType::IndexType IndexType;

      const IndexType row = threadIdx.x;
      if( row >= matrix->getRows() )
         return;

      IndexType* columnIndexes = getSharedMemory< IndexType >();
      RealType* valuesBase = ( RealType* ) & columnIndexes[ matrix->getColumns() ];
      RealType* values  = &valuesBase[ row * matrix->getColumns() ];

      columnIndexes[ row ] = row;

      for( int i = 0; i < matrix->getColumns(); i++ )
      {
         if( i == row )
            values[ i ] = 1.0 + 0.5 * i;
         else
            values[ i ] = 1.0;
      }
      matrix->setRowFast( row, columnIndexes, values, 10 );
   }

   template< typename MatrixType >
   __global__ void SparseTester__setRowFast_DenseTestCudaKernel2( MatrixType* matrix,
                                                                                 bool* testResult )
   {
      typedef typename MatrixType::RealType RealType;
      typedef typename MatrixType::IndexType IndexType;

      const IndexType row = threadIdx.x;
      if( row >= matrix->getRows() )
         return;

      IndexType* columnIndexes = getSharedMemory< IndexType >();
      RealType* valuesBase = ( RealType* ) & columnIndexes[ matrix->getColumns() ];
      RealType* values  = &valuesBase[ row * matrix->getColumns() ];

      columnIndexes[ row ] = row;

      for( int i = 0; i < matrix->getColumns(); i++ )
      {
            values[ i ] = row + i;
      }
      matrix->setRowFast( row, columnIndexes, values, 10 );
   }

   template< typename MatrixType >
   __global__ void SparseTester__setRowFast_LowerTriangularMatrixTestCudaKernel( MatrixType* matrix,
                                                                                          bool* testResult )
   {
      typedef typename MatrixType::RealType RealType;
      typedef typename MatrixType::IndexType IndexType;

      const IndexType row = threadIdx.x;
      if( row >= matrix->getRows() )
         return;

      IndexType* columnIndexes = getSharedMemory< IndexType >();
      RealType* valuesBase = ( RealType* ) & columnIndexes[ matrix->getColumns() ];
      RealType* values  = &valuesBase[ row * matrix->getColumns() ];

      columnIndexes[ row ] = row;

      for( int i = 0; i <= row; i++ )
      {
            values[ i ] = row + i;
      }
      matrix->setRowFast( row, columnIndexes, values, row + 1 );
   }

#endif


#endif

#endif /* SparseTESTER_H_ */
