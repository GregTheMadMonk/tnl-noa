/***************************************************************************
                          MatrixReader_impl.h  -  description
                             -------------------
    begin                : Dec 14, 2013
    copyright            : (C) 2013 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <iomanip>
#include <TNL/List.h>
#include <TNL/String.h>
#include <TNL/Vectors/Vector.h>
#include <TNL/TimerRT.h>

namespace TNL {
namespace Matrices {   

template< typename Matrix >
bool MatrixReader< Matrix >::readMtxFile( const String& fileName,
                                             Matrix& matrix,
                                             bool verbose )
{
   std::fstream file;
   file.open( fileName.getString(), std::ios::in );
   if( ! file )
   {
      std::cerr << "I am not able to open the file " << fileName << "." << std::endl;
      return false;
   }
   return readMtxFile( file, matrix, verbose );
}

template< typename Matrix >
bool MatrixReader< Matrix >::readMtxFile( std::istream& file,
                                             Matrix& matrix,
                                             bool verbose )
{
   return MatrixReaderDeviceDependentCode< typename Matrix::DeviceType >::readMtxFile( file, matrix, verbose );
}

template< typename Matrix >
bool MatrixReader< Matrix >::readMtxFileHostMatrix( std::istream& file,
                                                       Matrix& matrix,
                                                       typename Matrix::CompressedRowsLengthsVector& rowLengths,
                                                       bool verbose )
{
   IndexType rows, columns;
   bool symmetricMatrix( false );

   if( ! readMtxHeader( file, rows, columns, symmetricMatrix, verbose ) )
      return false;


   if( ! matrix.setDimensions( rows, columns ) ||
       ! rowLengths.setSize( rows ) )
   {
      std::cerr << "Not enough memory to allocate the sparse or the full matrix for testing." << std::endl;
      return false;
   }

   if( ! computeCompressedRowsLengthsFromMtxFile( file, rowLengths, columns, rows, symmetricMatrix, verbose ) )
      return false;

   if( ! matrix.setCompressedRowsLengths( rowLengths ) )
      return false;

   if( ! readMatrixElementsFromMtxFile( file, matrix, symmetricMatrix, verbose ) )
      return false;
   return true;
}

template< typename Matrix >
bool MatrixReader< Matrix >::verifyMtxFile( std::istream& file,
                                               const Matrix& matrix,
                                               bool verbose )
{
   bool symmetricMatrix( false );
   IndexType rows, columns;
   if( ! readMtxHeader( file, rows, columns, symmetricMatrix, false ) )
      return false;
   file.clear();
   file.seekg( 0, std::ios::beg );
   String line;
   bool dimensionsLine( false );
   IndexType processedElements( 0 );
   TimerRT timer;
   while( line.getLine( file ) )
   {
      if( line[ 0 ] == '%' ) continue;
      if( ! dimensionsLine )
      {
         dimensionsLine = true;
         continue;
      }
      IndexType row( 1 ), column( 1 );
      RealType value;
      if( ! parseMtxLineWithElement( line, row, column, value ) )
         return false;
      if( value != matrix.getElement( row-1, column-1 ) ||
          ( symmetricMatrix && value != matrix.getElement( column-1, row-1 ) ) )
      {
         std::cerr << "*** !!! VERIFICATION ERROR !!! *** " << std::endl
              << "The elements differ at " << row-1 << " row " << column-1 << " column." << std::endl
              << "The matrix value is " << matrix.getElement( row-1, column-1 )
              << " while the file value is " << value << "." << std::endl;
         return false;
      }
      processedElements++;
      if( symmetricMatrix && row != column )
         processedElements++;
      if( verbose )
        std::cout << " Verifying the matrix elements ... " << processedElements << " / " << matrix.getNumberOfMatrixElements() << "                       \r" << std::flush;
   }
   file.clear();
   long int fileSize = file.tellg();
   if( verbose )
     std::cout << " Verifying the matrix elements ... " << processedElements << " / " << matrix.getNumberOfMatrixElements()
           << " -> " << timer.getTime()
           << " sec. i.e. " << fileSize / ( timer.getTime() * ( 1 << 20 ))  << "MB/s." << std::endl;
   return true;
}

template< typename Matrix >
bool MatrixReader< Matrix >::findLineByElement( std::istream& file,
                                                   const IndexType& row,
                                                   const IndexType& column,
                                                   String& line,
                                                   IndexType& lineNumber )
{
   file.clear();
   file.seekg( 0, std::ios::beg );
   bool symmetricMatrix( false );
   bool dimensionsLine( false );
   lineNumber = 0;
   TimerRT timer;
   while( line.getLine( file ) )
   {
      lineNumber++;
      if( line[ 0 ] == '%' ) continue;
      if( ! dimensionsLine )
      {
         dimensionsLine = true;
         continue;
      }
      IndexType currentRow( 1 ), currentColumn( 1 );
      RealType value;
      if( ! parseMtxLineWithElement( line, currentRow, currentColumn, value ) )
         return false;
      if( ( currentRow == row + 1 && currentColumn == column + 1 ) ||
          ( symmetricMatrix && currentRow == column + 1 && currentColumn == row + 1 ) )
         return true;
   }
   return false;
}

template< typename Matrix >
bool MatrixReader< Matrix >::checkMtxHeader( const String& header,
                                                bool& symmetric )
{
   List< String > parsedLine;
   header.parse( parsedLine );
   if( parsedLine.getSize() < 5 )
      return false;
   if( parsedLine[ 0 ] != "%%MatrixMarket" )
      return false;
   if( parsedLine[ 1 ] != "matrix" )
   {
      std::cerr << "Error: 'matrix' expected in the header line (" << header << ")." << std::endl;
      return false;
   }
   if( parsedLine[ 2 ] != "coordinates" &&
       parsedLine[ 2 ] != "coordinate" )
   {
      std::cerr << "Error: Only 'coordinates' format is supported now, not " << parsedLine[ 2 ] << "." << std::endl;
      return false;
   }
   if( parsedLine[ 3 ] != "real" )
   {
      std::cerr << "Error: Only 'real' matrices are supported, not " << parsedLine[ 3 ] << "." << std::endl;
      return false;
   }
   if( parsedLine[ 4 ] != "general" )
   {
      if( parsedLine[ 4 ] == "symmetric" )
         symmetric = true;
      else
      {
         std::cerr << "Error: Only 'general' matrices are supported, not " << parsedLine[ 4 ] << "." << std::endl;
         return false;
      }
   }
   return true;
}

template< typename Matrix >
bool MatrixReader< Matrix >::readMtxHeader( std::istream& file,
                                               IndexType& rows,
                                               IndexType& columns,
                                               bool& symmetric,
                                               bool verbose )
{
   file.clear();
   file.seekg( 0, std::ios::beg );
   String line;
   bool headerParsed( false );
   List< String > parsedLine;
   while( true )
   {
      line.getLine( file );
      if( ! headerParsed )
      {
         headerParsed = checkMtxHeader( line, symmetric );
         if( ! headerParsed )
            return false;
         if( verbose && symmetric )
           std::cout << "The matrix is SYMMETRIC ... ";
         continue;
      }
      if( line[ 0 ] == '%' ) continue;
      if( ! headerParsed )
      {
         std::cerr << "Uknown format of the file. We expect line like this:" << std::endl;
         std::cerr << "%%MatrixMarket matrix coordinate real general" << std::endl;
         return false;
      }

      parsedLine.reset();
      line. parse( parsedLine );
      if( parsedLine. getSize() != 3 )
      {
         std::cerr << "Wrong number of parameters in the matrix header." << std::endl;
         return false;
      }
      rows = atoi( parsedLine[ 0 ]. getString() );
      columns = atoi( parsedLine[ 1 ]. getString() );
      if( verbose )
        std::cout << " The matrix has " << rows
              << " rows and " << columns << " columns. " << std::endl;

      if( rows <= 0 || columns <= 0 )
      {
         std::cerr << "Wrong parameters in the matrix header." << std::endl;
         return false;
      }
      return true;
   }
}

template< typename Matrix >
bool MatrixReader< Matrix >::computeCompressedRowsLengthsFromMtxFile( std::istream& file,
                                                              Vectors::Vector< int, Devices::Host, int >& rowLengths,
                                                              const int columns,
                                                              const int rows,
                                                              bool symmetricMatrix,
                                                              bool verbose )
{
   file.clear();
   file.seekg( 0,  std::ios::beg );
   rowLengths.setValue( 0 );
   String line;
   bool dimensionsLine( false );
   IndexType numberOfElements( 0 );
   TimerRT timer;
   while( line.getLine( file ) )
   {
      if( line[ 0 ] == '%' ) continue;
      if( ! dimensionsLine )
      {
         dimensionsLine = true;
         continue;
      }
      IndexType row( 1 ), column( 1 );
      RealType value;
      if( ! parseMtxLineWithElement( line, row, column, value ) )
         return false;
      numberOfElements++;
      if( column > columns || row > rows )
      {
         std::cerr << "There is an element at position " << row << ", " << column << " out of the matrix dimensions " << rows << " x " << columns << "." << std::endl;
         return false;
      }
      if( verbose )
        std::cout << " Counting the matrix elements ... " << numberOfElements / 1000 << " thousands      \r" << std::flush;
      rowLengths[ row - 1 ]++;
      if( rowLengths[ row - 1 ] > columns )
      {
         std::cerr << "There are more elements ( " << rowLengths[ row - 1 ] << " ) than the matrix columns ( " << columns << " ) at the row " << row << "." << std::endl;
         return false;
      }
      if( symmetricMatrix && row != column )
      {
         rowLengths[ column - 1 ]++;
         if( rowLengths[ column - 1 ] > columns )
         {
            std::cerr << "There are more elements ( " << rowLengths[ row - 1 ] << " ) than the matrix columns ( " << columns << " ) at the row " << column << " ." << std::endl;
            return false;
         }
      }
   }
   file.clear();
   long int fileSize = file.tellg();
   if( verbose )
     std::cout << " Counting the matrix elements ... " << numberOfElements / 1000
           << " thousands  -> " << timer.getTime()
           << " sec. i.e. " << fileSize / ( timer.getTime() * ( 1 << 20 ))  << "MB/s." << std::endl;
   return true;
}

template< typename Matrix >
bool MatrixReader< Matrix >::readMatrixElementsFromMtxFile( std::istream& file,
                                                               Matrix& matrix,
                                                               bool symmetricMatrix,
                                                               bool verbose )
{
   file.clear();
   file.seekg( 0,  std::ios::beg );
   String line;
   bool dimensionsLine( false );
   IndexType processedElements( 0 );
   TimerRT timer;
   while( line.getLine( file ) )
   {
      if( line[ 0 ] == '%' ) continue;
      if( ! dimensionsLine )
      {
         dimensionsLine = true;
         continue;
      }
      IndexType row( 1 ), column( 1 );
      RealType value;
      if( ! parseMtxLineWithElement( line, row, column, value ) )
         return false;
      matrix.setElement( row - 1, column - 1, value );
      processedElements++;
      if( symmetricMatrix && row != column )
      {
         matrix.setElement( column - 1, row - 1, value );
         processedElements++;
      }
      if( verbose )
        std::cout << " Reading the matrix elements ... " << processedElements << " / " << matrix.getNumberOfMatrixElements() << "                       \r" << std::flush;
   }
   file.clear();
   long int fileSize = file.tellg();
   if( verbose )
     std::cout << " Reading the matrix elements ... " << processedElements << " / " << matrix.getNumberOfMatrixElements()
              << " -> " << timer.getTime()
              << " sec. i.e. " << fileSize / ( timer.getTime() * ( 1 << 20 ))  << "MB/s." << std::endl;
   return true;
}

template< typename Matrix >
bool MatrixReader< Matrix >::parseMtxLineWithElement( const String& line,
                                                         IndexType& row,
                                                         IndexType& column,
                                                         RealType& value )
{
   List< String > parsedLine;
   line.parse( parsedLine );
   if( parsedLine.getSize() != 3 )
   {
      std::cerr << "Wrong number of parameters in the matrix row at line:" << line << std::endl;
      return false;
   }
   row = atoi( parsedLine[ 0 ].getString() );
   column = atoi( parsedLine[ 1 ].getString() );
   value = ( RealType ) atof( parsedLine[ 2 ].getString() );
   return true;
}

template<>
class MatrixReaderDeviceDependentCode< Devices::Host >
{
   public:

   template< typename Matrix >
   static bool readMtxFile( std::istream& file,
                            Matrix& matrix,
                            bool verbose )
   {
      typename Matrix::CompressedRowsLengthsVector rowLengths;
      return MatrixReader< Matrix >::readMtxFileHostMatrix( file, matrix, rowLengths, verbose );
   }
};

template<>
class MatrixReaderDeviceDependentCode< Devices::Cuda >
{
   public:

   template< typename Matrix >
   static bool readMtxFile( std::istream& file,
                            Matrix& matrix,
                            bool verbose )
   {
      typedef typename Matrix::HostType HostMatrixType;
      typedef typename HostMatrixType::CompressedRowsLengthsVector CompressedRowsLengthsVector;

      HostMatrixType hostMatrix;
      CompressedRowsLengthsVector rowLengthsVector;
      if( ! MatrixReader< HostMatrixType >::readMtxFileHostMatrix( file, hostMatrix, rowLengthsVector, verbose ) )
         return false;

      typename Matrix::CompressedRowsLengthsVector cudaCompressedRowsLengthsVector;
      cudaCompressedRowsLengthsVector.setLike( rowLengthsVector );
      cudaCompressedRowsLengthsVector = rowLengthsVector;
      if( ! matrix.copyFrom( hostMatrix, cudaCompressedRowsLengthsVector ) )
         return false;
      return true;
   }
};

} // namespace Matrices
} // namespace TNL
