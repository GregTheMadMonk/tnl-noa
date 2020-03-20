/***************************************************************************
                          LambdaMatrixTest.h -  description
                             -------------------
    begin                : Mar 18, 2020
    copyright            : (C) 2020 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#include <iostream>
#include <sstream>

#ifdef HAVE_GTEST
#include <gtest/gtest.h>

template< typename Matrix >
void test_Constructors()
{
   using RealType = typename Matrix::RealType;
   using DeviceType = typename Matrix::DeviceType;
   using IndexType = typename Matrix::IndexType;

   IndexType size = 5;
   auto rowLengths = [=] __cuda_callable__ ( const IndexType rows, const IndexType columns, const IndexType rowIdx ) -> IndexType { return 1; };
   auto matrixElements = [=] __cuda_callable__ ( const IndexType rows, const IndexType columns, const IndexType rowIdx, const IndexType localIdx, IndexType& columnIdx, RealType& value ) {
         columnIdx = rowIdx;
         value =  1.0;
   };

   using MatrixType = decltype( TNL::Matrices::LambdaMatrixFactory< RealType, DeviceType, IndexType >::create( matrixElements, rowLengths ) );

   MatrixType m( size, size, matrixElements, rowLengths );

   EXPECT_EQ( m.getRows(), size );
   EXPECT_EQ( m.getColumns(), size );
}

template< typename Matrix >
void test_SetDimensions()
{
   using RealType = typename Matrix::RealType;
   using DeviceType = typename Matrix::DeviceType;
   using IndexType = typename Matrix::IndexType;

   IndexType size = 5;
   auto rowLengths = [=] __cuda_callable__ ( const IndexType rows, const IndexType columns, const IndexType rowIdx ) -> IndexType { return 1; };
   auto matrixElements = [=] __cuda_callable__ ( const IndexType rows, const IndexType columns, const IndexType rowIdx, const IndexType localIdx, IndexType& columnIdx, RealType& value ) {
         columnIdx = rowIdx;
         value =  1.0;
   };

   using MatrixType = decltype( TNL::Matrices::LambdaMatrixFactory< RealType, DeviceType, IndexType >::create( matrixElements, rowLengths ) );

   MatrixType m( size, size, matrixElements, rowLengths );

   EXPECT_EQ( m.getRows(), size );
   EXPECT_EQ( m.getColumns(), size );

   m.setDimensions( 10, 10 );
   EXPECT_EQ( m.getRows(), 10 );
   EXPECT_EQ( m.getColumns(), 10 );

}

template< typename Matrix >
void test_GetCompressedRowLengths()
{
   using RealType = typename Matrix::RealType;
   using DeviceType = typename Matrix::DeviceType;
   using IndexType = typename Matrix::IndexType;

   IndexType size = 5;
   auto rowLengths = [=] __cuda_callable__ ( const IndexType rows, const IndexType columns, const IndexType rowIdx ) -> IndexType {
      if( rowIdx == 0 || rowIdx == size - 1 )
         return 1;
      return 3;
   };

   auto matrixElements = [=] __cuda_callable__ ( const IndexType rows, const IndexType columns, const IndexType rowIdx, const IndexType localIdx, IndexType& columnIdx, RealType& value ) {
      if( rowIdx == 0 || rowIdx == size -1 )
      {
         columnIdx = rowIdx;
         value =  1.0;
      }
      else
      {
         columnIdx = rowIdx + localIdx - 1;
         value = ( columnIdx == rowIdx ) ? -2.0 : 1.0;
      }
   };

   using MatrixType = decltype( TNL::Matrices::LambdaMatrixFactory< RealType, DeviceType, IndexType >::create( matrixElements, rowLengths ) );

   MatrixType m( size, size, matrixElements, rowLengths );
   TNL::Containers::Vector< IndexType > correctRowLengths{ 1, 3, 3, 3, 1 };
   TNL::Containers::Vector< IndexType > rowLengthsVector;
   m.getCompressedRowLengths( rowLengthsVector );
   for( int i = 0; i < size; i++ )
      EXPECT_EQ( correctRowLengths[ i ], rowLengthsVector[ i ] );
}

template< typename Matrix >
void test_GetElement()
{
   using RealType = typename Matrix::RealType;
   using DeviceType = typename Matrix::DeviceType;
   using IndexType = typename Matrix::IndexType;

   IndexType size = 5;
   auto rowLengths = [=] __cuda_callable__ ( const IndexType rows, const IndexType columns, const IndexType rowIdx ) -> IndexType {
      if( rowIdx == 0 || rowIdx == size - 1 )
         return 1;
      return 3;
   };

   auto matrixElements = [=] __cuda_callable__ ( const IndexType rows, const IndexType columns, const IndexType rowIdx, const IndexType localIdx, IndexType& columnIdx, RealType& value ) {
      if( rowIdx == 0 || rowIdx == size -1 )
      {
         columnIdx = rowIdx;
         value =  1.0;
      }
      else
      {
         columnIdx = rowIdx + localIdx - 1;
         value = ( columnIdx == rowIdx ) ? -2.0 : 1.0;
      }
   };

   using MatrixType = decltype( TNL::Matrices::LambdaMatrixFactory< RealType, DeviceType, IndexType >::create( matrixElements, rowLengths ) );

   MatrixType m( size, size, matrixElements, rowLengths );
   EXPECT_EQ( m.getElement( 0, 0 ),  1.0 );
   EXPECT_EQ( m.getElement( 0, 1 ),  0.0 );
   EXPECT_EQ( m.getElement( 0, 2 ),  0.0 );
   EXPECT_EQ( m.getElement( 0, 3 ),  0.0 );
   EXPECT_EQ( m.getElement( 0, 4 ),  0.0 );

   EXPECT_EQ( m.getElement( 1, 0 ),  1.0 );
   EXPECT_EQ( m.getElement( 1, 1 ), -2.0 );
   EXPECT_EQ( m.getElement( 1, 2 ),  1.0 );
   EXPECT_EQ( m.getElement( 1, 3 ),  0.0 );
   EXPECT_EQ( m.getElement( 1, 4 ),  0.0 );

   EXPECT_EQ( m.getElement( 2, 0 ),  0.0 );
   EXPECT_EQ( m.getElement( 2, 1 ),  1.0 );
   EXPECT_EQ( m.getElement( 2, 2 ), -2.0 );
   EXPECT_EQ( m.getElement( 2, 3 ),  1.0 );
   EXPECT_EQ( m.getElement( 2, 4 ),  0.0 );

   EXPECT_EQ( m.getElement( 3, 0 ),  0.0 );
   EXPECT_EQ( m.getElement( 3, 1 ),  0.0 );
   EXPECT_EQ( m.getElement( 3, 2 ),  1.0 );
   EXPECT_EQ( m.getElement( 3, 3 ), -2.0 );
   EXPECT_EQ( m.getElement( 3, 4 ),  1.0 );

   EXPECT_EQ( m.getElement( 4, 0 ),  0.0 );
   EXPECT_EQ( m.getElement( 4, 1 ),  0.0 );
   EXPECT_EQ( m.getElement( 4, 2 ),  0.0 );
   EXPECT_EQ( m.getElement( 4, 3 ),  0.0 );
   EXPECT_EQ( m.getElement( 4, 4 ),  1.0 );
}

template< typename Matrix >
void test_VectorProduct()
{
   using RealType = typename Matrix::RealType;
   using DeviceType = typename Matrix::DeviceType;
   using IndexType = typename Matrix::IndexType;

   IndexType size = 5;
   auto rowLengths = [=] __cuda_callable__ ( const IndexType rows, const IndexType columns, const IndexType rowIdx ) -> IndexType {
      if( rowIdx == 0 || rowIdx == size - 1 )
         return 1;
      return 3;
   };

   auto matrixElements = [=] __cuda_callable__ ( const IndexType rows, const IndexType columns, const IndexType rowIdx, const IndexType localIdx, IndexType& columnIdx, RealType& value ) {
      if( rowIdx == 0 || rowIdx == size -1 )
      {
         columnIdx = rowIdx;
         value =  1.0;
      }
      else
      {
         columnIdx = rowIdx + localIdx - 1;
         value = ( columnIdx == rowIdx ) ? -2.0 : 1.0;
      }
   };

   using MatrixType = decltype( TNL::Matrices::LambdaMatrixFactory< RealType, DeviceType, IndexType >::create( matrixElements, rowLengths ) );

   MatrixType A( size, size, matrixElements, rowLengths );
   TNL::Containers::Vector< RealType, DeviceType, IndexType > x( size, 1.0 ), b( size, 5.0 );
   A.vectorProduct( x, b );
   EXPECT_EQ( b.getElement( 0 ),  1.0 );
   EXPECT_EQ( b.getElement( 1 ),  0.0 );
   EXPECT_EQ( b.getElement( 2 ),  0.0 );
   EXPECT_EQ( b.getElement( 3 ),  0.0 );
   EXPECT_EQ( b.getElement( 4 ),  1.0 );
}

template< typename Matrix >
void test_RowsReduction()
{
   using RealType = typename Matrix::RealType;
   using DeviceType = typename Matrix::DeviceType;
   using IndexType = typename Matrix::IndexType;

   IndexType size = 5;
   auto rowLengths = [=] __cuda_callable__ ( const IndexType rows, const IndexType columns, const IndexType rowIdx ) -> IndexType {
      if( rowIdx == 0 || rowIdx == size - 1 )
         return 1;
      return 3;
   };

   auto matrixElements = [=] __cuda_callable__ ( const IndexType rows, const IndexType columns, const IndexType rowIdx, const IndexType localIdx, IndexType& columnIdx, RealType& value ) {
      if( rowIdx == 0 || rowIdx == size -1 )
      {
         columnIdx = rowIdx;
         value =  1.0;
      }
      else
      {
         columnIdx = rowIdx + localIdx - 1;
         value = ( columnIdx == rowIdx ) ? -2.0 : 1.0;
      }
   };

   using MatrixType = decltype( TNL::Matrices::LambdaMatrixFactory< RealType, DeviceType, IndexType >::create( matrixElements, rowLengths ) );

   MatrixType A( size, size, matrixElements, rowLengths );
   TNL::Containers::Vector< RealType, DeviceType, IndexType > v( size, -1.0 );
   auto vView = v.getView();

   auto fetch = [=] __cuda_callable__ ( IndexType row, IndexType localIdx, IndexType columnIdx, const RealType& value ) mutable -> RealType {
      return value;
   };
   auto reduce = [] __cuda_callable__ ( RealType& sum, const RealType& value ) {
      sum += value;
   };
   auto keep = [=] __cuda_callable__ ( IndexType row, const RealType& value ) mutable {
      vView[ row ] = value;
   };
   A.allRowsReduction( fetch, reduce, keep, 0.0 );

   EXPECT_EQ( v.getElement( 0 ),  1.0 );
   EXPECT_EQ( v.getElement( 1 ),  0.0 );
   EXPECT_EQ( v.getElement( 2 ),  0.0 );
   EXPECT_EQ( v.getElement( 3 ),  0.0 );
   EXPECT_EQ( v.getElement( 4 ),  1.0 );
}

template< typename Matrix >
void test_Print()
{
   using RealType = typename Matrix::RealType;
   using DeviceType = typename Matrix::DeviceType;
   using IndexType = typename Matrix::IndexType;

   IndexType size = 5;
   auto rowLengths = [=] __cuda_callable__ ( const IndexType rows, const IndexType columns, const IndexType rowIdx ) -> IndexType {
      if( rowIdx == 0 || rowIdx == size - 1 )
         return 1;
      return 3;
   };

   auto matrixElements = [=] __cuda_callable__ ( const IndexType rows, const IndexType columns, const IndexType rowIdx, const IndexType localIdx, IndexType& columnIdx, RealType& value ) {
      if( rowIdx == 0 || rowIdx == size -1 )
      {
         columnIdx = rowIdx;
         value =  1.0;
      }
      else
      {
         columnIdx = rowIdx + localIdx - 1;
         value = ( columnIdx == rowIdx ) ? -2.0 : 1.0;
      }
   };

   using MatrixType = decltype( TNL::Matrices::LambdaMatrixFactory< RealType, DeviceType, IndexType >::create( matrixElements, rowLengths ) );

   MatrixType m( size, size, matrixElements, rowLengths );

   std::stringstream printed;
   std::stringstream couted;

   //change the underlying buffer and save the old buffer
   auto old_buf = std::cout.rdbuf(printed.rdbuf());

   
   m.print( std::cout ); //all the std::cout goes to ss

   std::cout.rdbuf(old_buf); //reset

   couted << "Row: 0 ->  Col:0->1\t\n"
             "Row: 1 ->  Col:0->1	 Col:1->-2	 Col:2->1\t\n"
             "Row: 2 ->  Col:1->1	 Col:2->-2	 Col:3->1\t\n"
             "Row: 3 ->  Col:2->1	 Col:3->-2	 Col:4->1\t\n"
             "Row: 4 ->  Col:4->1\t\n";

   EXPECT_EQ( printed.str(), couted.str() );
}


#endif // HAVE_GTEST
