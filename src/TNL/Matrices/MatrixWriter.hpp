/***************************************************************************
                          MatrixWriter_impl.h  -  description
                             -------------------
    begin                : Dec 18, 2013
    copyright            : (C) 2013 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <iomanip>
#include <TNL/Matrices/MatrixWriter.h>

namespace TNL {
namespace Matrices {

template< typename Matrix, typename Device >
void
MatrixWriter< Matrix, Device >::
writeToGnuplot( const TNL::String& fileName,
                const Matrix& matrix,
                bool verbose )
{
   HostMatrix hostMatrix;
   hostMatrix = matrix;
   MatrixWriter< HostMatrix >::writeToGnuplot( fileName, hostMatrix, verbose );
}

template< typename Matrix, typename Device >
void
MatrixWriter< Matrix, Device >::
writeToGnuplot( std::ostream& str,
                const Matrix& matrix,
                bool verbose )
{
   HostMatrix hostMatrix;
   hostMatrix = matrix;
   MatrixWriter< HostMatrix >::writeToGnuplot( str, hostMatrix, verbose );
}

template< typename Matrix, typename Device >
void
MatrixWriter< Matrix, Device >::
writeToMtx( const TNL::String& fileName,
            const Matrix& matrix,
            bool verbose )
{
   HostMatrix hostMatrix;
   hostMatrix = matrix;
   MatrixWriter< HostMatrix >::writeToMtx( fileName, hostMatrix, verbose );
}

template< typename Matrix, typename Device >
void
MatrixWriter< Matrix, Device >::
writeToMtx( std::ostream& str,
            const Matrix& matrix,
            bool verbose )
{
   HostMatrix hostMatrix;
   hostMatrix = matrix;
   MatrixWriter< HostMatrix >::writeToMtx( str, hostMatrix, verbose );
}

template< typename Matrix, typename Device >
void
MatrixWriter< Matrix, Device >::
writeToEps( const TNL::String& fileName,
            const Matrix& matrix,
            bool verbose )
{
   HostMatrix hostMatrix;
   hostMatrix = matrix;
   MatrixWriter< HostMatrix >::writeToEps( fileName, hostMatrix, verbose );
}

template< typename Matrix, typename Device >
void
MatrixWriter< Matrix, Device >::
writeToEps( std::ostream& str,
            const Matrix& matrix,
            bool verbose )
{
   HostMatrix hostMatrix;
   hostMatrix = matrix;
   MatrixWriter< HostMatrix >::writeToEps( str, hostMatrix, verbose );
}

/**
 * MatrixWriter specialization for TNL::Devices::Host.
 */

template< typename Matrix >
void
MatrixWriter< Matrix, TNL::Devices::Host >::
writeToGnuplot( const TNL::String& fileName,
                const Matrix& matrix,
                bool verbose )
{
   std::fstream str;
   str.open( fileName.getString(), std::ios::out );
   MatrixWriter< Matrix >::writeToGnuplot( str, matrix, verbose );
}

template< typename Matrix >
void
MatrixWriter< Matrix, TNL::Devices::Host >::
writeToGnuplot( std::ostream& str,
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
MatrixWriter< Matrix, TNL::Devices::Host >::
writeToMtx( const TNL::String& fileName,
            const Matrix& matrix,
            bool verbose )
{
   std::fstream str;
   str.open( fileName.getString(), std::ios::out );
   MatrixWriter< Matrix >::writeToMtx( str, matrix, verbose );
}

template< typename Matrix >
void
MatrixWriter< Matrix, TNL::Devices::Host >::
writeToMtx( std::ostream& str,
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
   auto f = [=] __cuda_callable__ ( IndexType rowIdx, IndexType localIdx, IndexType columnIdx, RealType value, bool& compute ) mutable {
      if( value != 0 )
      {
         *str_ptr << std::setw( 9 ) << rowIdx + 1 << std::setw( 9 ) << columnIdx + 1 << std::setw( 12 ) << value << std::endl;
         if( verbose )
            *cout_ptr << "Drawing the row " << rowIdx << "      \r" << std::flush;
      }
   };
   matrix.forAllRows( f );
}

template< typename Matrix >
void
MatrixWriter< Matrix, TNL::Devices::Host >::
writeToEps( const TNL::String& fileName,
            const Matrix& matrix,
            bool verbose )
{
   std::fstream str;
   str.open( fileName.getString(), std::ios::out );
   MatrixWriter< Matrix >::writeToEps( str, matrix, verbose );
}

template< typename Matrix >
void
MatrixWriter< Matrix, TNL::Devices::Host >::
writeToEps( std::ostream& str,
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
MatrixWriter< Matrix, TNL::Devices::Host >::
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
MatrixWriter< Matrix, TNL::Devices::Host >::
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
} // namespace TNL
