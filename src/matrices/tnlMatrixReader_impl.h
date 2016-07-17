/***************************************************************************
                          tnlMatrixReader_impl.h  -  description
                             -------------------
    begin                : Dec 14, 2013
    copyright            : (C) 2013 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <iomanip>
#include <core/tnlList.h>
#include <core/tnlString.h>
#include <core/vectors/tnlVector.h>
#include <core/tnlTimerRT.h>

namespace TNL {

template< typename Matrix >
bool tnlMatrixReader< Matrix >::readMtxFile( const tnlString& fileName,
                                             Matrix& matrix,
                                             bool verbose )
{
   fstream file;
   file.open( fileName.getString(), ios::in );
   if( ! file )
   {
      cerr << "I am not able to open the file " << fileName << "." << endl;
      return false;
   }
   return readMtxFile( file, matrix, verbose );
}

template< typename Matrix >
bool tnlMatrixReader< Matrix >::readMtxFile( std::istream& file,
                                             Matrix& matrix,
                                             bool verbose )
{
   return tnlMatrixReaderDeviceDependentCode< typename Matrix::DeviceType >::readMtxFile( file, matrix, verbose );
}

template< typename Matrix >
bool tnlMatrixReader< Matrix >::readMtxFileHostMatrix( std::istream& file,
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
      cerr << "Not enough memory to allocate the sparse or the full matrix for testing." << endl;
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
bool tnlMatrixReader< Matrix >::verifyMtxFile( std::istream& file,
                                               const Matrix& matrix,
                                               bool verbose )
{
   bool symmetricMatrix( false );
   IndexType rows, columns;
   if( ! readMtxHeader( file, rows, columns, symmetricMatrix, false ) )
      return false;
   file.clear();
   file.seekg( 0, ios::beg );
   tnlString line;
   bool dimensionsLine( false );
   IndexType processedElements( 0 );
   tnlTimerRT timer;
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
         cerr << "*** !!! VERIFICATION ERROR !!! *** " << endl
              << "The elements differ at " << row-1 << " row " << column-1 << " column." << endl
              << "The matrix value is " << matrix.getElement( row-1, column-1 )
              << " while the file value is " << value << "." << endl;
         return false;
      }
      processedElements++;
      if( symmetricMatrix && row != column )
         processedElements++;
      if( verbose )
         cout << " Verifying the matrix elements ... " << processedElements << " / " << matrix.getNumberOfMatrixElements() << "                       \r" << flush;
   }
   file.clear();
   long int fileSize = file.tellg();
   if( verbose )
      cout << " Verifying the matrix elements ... " << processedElements << " / " << matrix.getNumberOfMatrixElements()
           << " -> " << timer.getTime()
           << " sec. i.e. " << fileSize / ( timer.getTime() * ( 1 << 20 ))  << "MB/s." << endl;
   return true;
}

template< typename Matrix >
bool tnlMatrixReader< Matrix >::findLineByElement( std::istream& file,
                                                   const IndexType& row,
                                                   const IndexType& column,
                                                   tnlString& line,
                                                   IndexType& lineNumber )
{
   file.clear();
   file.seekg( 0, ios::beg );
   bool symmetricMatrix( false );
   bool dimensionsLine( false );
   lineNumber = 0;
   tnlTimerRT timer;
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
bool tnlMatrixReader< Matrix >::checkMtxHeader( const tnlString& header,
                                                bool& symmetric )
{
   tnlList< tnlString > parsedLine;
   header.parse( parsedLine );
   if( parsedLine.getSize() < 5 )
      return false;
   if( parsedLine[ 0 ] != "%%MatrixMarket" )
      return false;
   if( parsedLine[ 1 ] != "matrix" )
   {
      cerr << "Error: 'matrix' expected in the header line (" << header << ")." << endl;
      return false;
   }
   if( parsedLine[ 2 ] != "coordinates" &&
       parsedLine[ 2 ] != "coordinate" )
   {
      cerr << "Error: Only 'coordinates' format is supported now, not " << parsedLine[ 2 ] << "." << endl;
      return false;
   }
   if( parsedLine[ 3 ] != "real" )
   {
      cerr << "Error: Only 'real' matrices are supported, not " << parsedLine[ 3 ] << "." << endl;
      return false;
   }
   if( parsedLine[ 4 ] != "general" )
   {
      if( parsedLine[ 4 ] == "symmetric" )
         symmetric = true;
      else
      {
         cerr << "Error: Only 'general' matrices are supported, not " << parsedLine[ 4 ] << "." << endl;
         return false;
      }
   }
   return true;
}

template< typename Matrix >
bool tnlMatrixReader< Matrix >::readMtxHeader( std::istream& file,
                                               IndexType& rows,
                                               IndexType& columns,
                                               bool& symmetric,
                                               bool verbose )
{
   file.clear();
   file.seekg( 0, ios::beg );
   tnlString line;
   bool headerParsed( false );
   tnlList< tnlString > parsedLine;
   while( true )
   {
      line.getLine( file );
      if( ! headerParsed )
      {
         headerParsed = checkMtxHeader( line, symmetric );
         if( ! headerParsed )
            return false;
         if( verbose && symmetric )
            cout << "The matrix is SYMMETRIC ... ";
         continue;
      }
      if( line[ 0 ] == '%' ) continue;
      if( ! headerParsed )
      {
         cerr << "Uknown format of the file. We expect line like this:" << endl;
         cerr << "%%MatrixMarket matrix coordinate real general" << endl;
         return false;
      }

      parsedLine.reset();
      line. parse( parsedLine );
      if( parsedLine. getSize() != 3 )
      {
         cerr << "Wrong number of parameters in the matrix header." << endl;
         return false;
      }
      rows = atoi( parsedLine[ 0 ]. getString() );
      columns = atoi( parsedLine[ 1 ]. getString() );
      if( verbose )
         cout << " The matrix has " << rows
              << " rows and " << columns << " columns. " << endl;

      if( rows <= 0 || columns <= 0 )
      {
         cerr << "Wrong parameters in the matrix header." << endl;
         return false;
      }
      return true;
   }
}

template< typename Matrix >
bool tnlMatrixReader< Matrix >::computeCompressedRowsLengthsFromMtxFile( std::istream& file,
                                                              tnlVector< int, tnlHost, int >& rowLengths,
                                                              const int columns,
                                                              const int rows,
                                                              bool symmetricMatrix,
                                                              bool verbose )
{
   file.clear();
   file.seekg( 0,  ios::beg );
   rowLengths.setValue( 0 );
   tnlString line;
   bool dimensionsLine( false );
   IndexType numberOfElements( 0 );
   tnlTimerRT timer;
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
         cerr << "There is an element at position " << row << ", " << column << " out of the matrix dimensions " << rows << " x " << columns << "." << endl;
         return false;
      }
      if( verbose )
         cout << " Counting the matrix elements ... " << numberOfElements / 1000 << " thousands      \r" << flush;
      rowLengths[ row - 1 ]++;
      if( rowLengths[ row - 1 ] > columns )
      {
         cerr << "There are more elements ( " << rowLengths[ row - 1 ] << " ) than the matrix columns ( " << columns << " ) at the row " << row << "." << endl;
         return false;
      }
      if( symmetricMatrix && row != column )
      {
         rowLengths[ column - 1 ]++;
         if( rowLengths[ column - 1 ] > columns )
         {
            cerr << "There are more elements ( " << rowLengths[ row - 1 ] << " ) than the matrix columns ( " << columns << " ) at the row " << column << " ." << endl;
            return false;
         }
      }
   }
   file.clear();
   long int fileSize = file.tellg();
   if( verbose )
      cout << " Counting the matrix elements ... " << numberOfElements / 1000
           << " thousands  -> " << timer.getTime()
           << " sec. i.e. " << fileSize / ( timer.getTime() * ( 1 << 20 ))  << "MB/s." << endl;
   return true;
}

template< typename Matrix >
bool tnlMatrixReader< Matrix >::readMatrixElementsFromMtxFile( std::istream& file,
                                                               Matrix& matrix,
                                                               bool symmetricMatrix,
                                                               bool verbose )
{
   file.clear();
   file.seekg( 0,  ios::beg );
   tnlString line;
   bool dimensionsLine( false );
   IndexType processedElements( 0 );
   tnlTimerRT timer;
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
         cout << " Reading the matrix elements ... " << processedElements << " / " << matrix.getNumberOfMatrixElements() << "                       \r" << flush;
   }
   file.clear();
   long int fileSize = file.tellg();
   if( verbose )
      cout << " Reading the matrix elements ... " << processedElements << " / " << matrix.getNumberOfMatrixElements()
              << " -> " << timer.getTime()
              << " sec. i.e. " << fileSize / ( timer.getTime() * ( 1 << 20 ))  << "MB/s." << endl;
   return true;
}

template< typename Matrix >
bool tnlMatrixReader< Matrix >::parseMtxLineWithElement( const tnlString& line,
                                                         IndexType& row,
                                                         IndexType& column,
                                                         RealType& value )
{
   tnlList< tnlString > parsedLine;
   line.parse( parsedLine );
   if( parsedLine.getSize() != 3 )
   {
      cerr << "Wrong number of parameters in the matrix row at line:" << line << endl;
      return false;
   }
   row = atoi( parsedLine[ 0 ].getString() );
   column = atoi( parsedLine[ 1 ].getString() );
   value = ( RealType ) atof( parsedLine[ 2 ].getString() );
   return true;
}

template<>
class tnlMatrixReaderDeviceDependentCode< tnlHost >
{
   public:

   template< typename Matrix >
   static bool readMtxFile( std::istream& file,
                            Matrix& matrix,
                            bool verbose )
   {
      typename Matrix::CompressedRowsLengthsVector rowLengths;
      return tnlMatrixReader< Matrix >::readMtxFileHostMatrix( file, matrix, rowLengths, verbose );
   }
};

template<>
class tnlMatrixReaderDeviceDependentCode< tnlCuda >
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
      if( ! tnlMatrixReader< HostMatrixType >::readMtxFileHostMatrix( file, hostMatrix, rowLengthsVector, verbose ) )
         return false;

      typename Matrix::CompressedRowsLengthsVector cudaCompressedRowsLengthsVector;
      cudaCompressedRowsLengthsVector.setLike( rowLengthsVector );
      cudaCompressedRowsLengthsVector = rowLengthsVector;
      if( ! matrix.copyFrom( hostMatrix, cudaCompressedRowsLengthsVector ) )
         return false;
      return true;
   }
};

} // namespace TNL
