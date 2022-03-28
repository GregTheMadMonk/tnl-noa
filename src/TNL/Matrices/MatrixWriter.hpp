// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <iomanip>
#include <noa/3rdparty/tnl-noa/src/TNL/Matrices/MatrixWriter.h>

namespace noa::TNL {
namespace Matrices {

template< typename Matrix, typename Device >
void
MatrixWriter< Matrix, Device >::
writeGnuplot( const noa::TNL::String& fileName,
              const Matrix& matrix,
              bool verbose )
{
   HostMatrix hostMatrix;
   hostMatrix = matrix;
   MatrixWriter< HostMatrix >::writeGnuplot( fileName, hostMatrix, verbose );
}

template< typename Matrix, typename Device >
void
MatrixWriter< Matrix, Device >::
writeGnuplot( std::ostream& str,
              const Matrix& matrix,
              bool verbose )
{
   HostMatrix hostMatrix;
   hostMatrix = matrix;
   MatrixWriter< HostMatrix >::writeGnuplot( str, hostMatrix, verbose );
}

template< typename Matrix, typename Device >
void
MatrixWriter< Matrix, Device >::
writeMtx( const noa::TNL::String& fileName,
          const Matrix& matrix,
          bool verbose )
{
   HostMatrix hostMatrix;
   hostMatrix = matrix;
   MatrixWriter< HostMatrix >::writeMtx( fileName, hostMatrix, verbose );
}

template< typename Matrix, typename Device >
void
MatrixWriter< Matrix, Device >::
writeMtx( std::ostream& str,
          const Matrix& matrix,
          bool verbose )
{
   HostMatrix hostMatrix;
   hostMatrix = matrix;
   MatrixWriter< HostMatrix >::writeMtx( str, hostMatrix, verbose );
}

template< typename Matrix, typename Device >
void
MatrixWriter< Matrix, Device >::
writeEps( const noa::TNL::String& fileName,
          const Matrix& matrix,
          bool verbose )
{
   HostMatrix hostMatrix;
   hostMatrix = matrix;
   MatrixWriter< HostMatrix >::writeEps( fileName, hostMatrix, verbose );
}

template< typename Matrix, typename Device >
void
MatrixWriter< Matrix, Device >::
writeEps( std::ostream& str,
          const Matrix& matrix,
          bool verbose )
{
   HostMatrix hostMatrix;
   hostMatrix = matrix;
   MatrixWriter< HostMatrix >::writeEps( str, hostMatrix, verbose );
}

/**
 * MatrixWriter specialization for noa::TNL::Devices::Host.
 */
template< typename Matrix >
void
MatrixWriter< Matrix, noa::TNL::Devices::Host >::
writeGnuplot( const noa::TNL::String& fileName,
              const Matrix& matrix,
              bool verbose )
{
   std::fstream str;
   str.open( fileName.getString(), std::ios::out );
   MatrixWriter< Matrix >::writeGnuplot( str, matrix, verbose );
}

template< typename Matrix >
void
MatrixWriter< Matrix, noa::TNL::Devices::Host >::
writeGnuplot( std::ostream& str,
              const Matrix& matrix,
              bool verbose )
{
   str << "#  This file was generated by TNL (www.tnl-project.org)" << std::endl;
   for( IndexType row = 0; row < matrix.getRows(); row ++ )
   {
      for( IndexType column = 0; column < matrix.getColumns(); column ++ )
      {
         RealType elementValue = matrix.getElement( row, column );
         if(  elementValue != ( RealType ) 0.0 )
            str << column << " " << row << " " << elementValue << "\n";
      }
      if( verbose )
        std::cout << "Drawing the row " << row << "      \r" << std::flush;
   }
   if( verbose )
     std::cout << std::endl;
}

template< typename Matrix >
void
MatrixWriter< Matrix, noa::TNL::Devices::Host >::
writeMtx( const noa::TNL::String& fileName,
          const Matrix& matrix,
          bool verbose )
{
   std::fstream str;
   str.open( fileName.getString(), std::ios::out );
   MatrixWriter< Matrix >::writeMtx( str, matrix, verbose );
}

template< typename Matrix >
void
MatrixWriter< Matrix, noa::TNL::Devices::Host >::
writeMtx( std::ostream& str,
          const Matrix& matrix,
          bool verbose )
{
   str << "%%MatrixMarket matrix coordinate real general" << std::endl;
   str << "%%" << std::endl;
   str << "%% This file was generated by TNL (www.tnl-project.org)" << std::endl;
   str << "%%" << std::setw( 9 ) << " ROWS " << std::setw( 9 ) << " COLUMNS " << std::setw( 12 ) << " ELEMENTS " << std::endl;
   str << std::setw( 9 ) << matrix.getRows() << " " << std::setw( 9 ) << matrix.getColumns() << " " << std::setw( 12 ) << matrix.getNonzeroElementsCount() << std::endl;
   std::ostream* str_ptr = &str;
   auto cout_ptr = &std::cout;
   auto f = [=] __cuda_callable__ ( const typename Matrix::ConstRowView& row ) mutable {
      auto rowIdx = row.getRowIndex();
      for( IndexType localIdx = 0; localIdx < row.getSize(); localIdx++ )
      {
         IndexType columnIdx = row.getColumnIndex( localIdx );
         RealType value = row.getValue( localIdx );
         if( value != 0 )
         {
            *str_ptr << std::setw( 9 ) << rowIdx + 1 << std::setw( 9 ) << columnIdx + 1 << std::setw( 12 ) << value << std::endl;
            if( verbose )
               *cout_ptr << "Drawing the row " << rowIdx << "      \r" << std::flush;
         }
      }
   };
   matrix.sequentialForAllRows( f );
}

template< typename Matrix >
void
MatrixWriter< Matrix, noa::TNL::Devices::Host >::
writeEps( const noa::TNL::String& fileName,
            const Matrix& matrix,
            bool verbose )
{
   std::fstream str;
   str.open( fileName.getString(), std::ios::out );
   MatrixWriter< Matrix >::writeEps( str, matrix, verbose );
}

template< typename Matrix >
void
MatrixWriter< Matrix, noa::TNL::Devices::Host >::
writeEps( std::ostream& str,
            const Matrix& matrix,
            bool verbose )
{
   const int elementSize = 10;
   writeEpsHeader( str, matrix, elementSize );
   writeEpsBody( str, matrix, elementSize, verbose );

   str << "showpage" << std::endl;
   str << "%%EOF" << std::endl;

   if( verbose )
     std::cout << std::endl;
}

template< typename Matrix >
void
MatrixWriter< Matrix, noa::TNL::Devices::Host >::
writeEpsHeader( std::ostream& str,
                const Matrix& matrix,
                const int elementSize )
{
   const double scale = elementSize * max( matrix.getRows(), matrix.getColumns() );
   str << "%!PS-Adobe-2.0 EPSF-2.0" << std::endl;
   str << "%%BoundingBox: 0 0 " << scale << " " << scale << std::endl;
   str << "%%Creator: TNL" << std::endl;
   str << "%%LanguageLevel: 2" << std::endl;
   str << "%%EndComments" << std::endl << std::endl;
   str << "0 " << scale << " translate" << std::endl;
}

template< typename Matrix >
void
MatrixWriter< Matrix, noa::TNL::Devices::Host >::
writeEpsBody( std::ostream& str,
              const Matrix& matrix,
              const int elementSize,
              bool verbose )
{
   IndexType lastRow( 0 ), lastColumn( 0 );
   for( IndexType row = 0; row < matrix.getRows(); row ++ )
   {
      for( IndexType column = 0; column < matrix.getColumns(); column ++ )
      {
         RealType elementValue = matrix.getElement( row, column );
         if( elementValue != ( RealType ) 0.0 )
         {
            str << ( column - lastColumn ) * elementSize
                << " " << -( row - lastRow ) * elementSize
                << " translate newpath 0 0 " << elementSize << " " << elementSize << " rectstroke\n";
            lastColumn = column;
            lastRow = row;
         }
      }
      if( verbose )
        std::cout << "Drawing the row " << row << "      \r" << std::flush;
   }
}


} // namespace Matrices
} // namespace noa::TNL
