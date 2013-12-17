/***************************************************************************
                          tnlMatrixReader_impl.h  -  description
                             -------------------
    begin                : Dec 14, 2013
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

#ifndef TNLMATRIXREADER_IMPL_H_
#define TNLMATRIXREADER_IMPL_H_

#include <iomanip>
#include <core/tnlString.h>
#include <core/vectors/tnlVector.h>

using namespace std;

template< typename Matrix >
bool tnlMatrixReader::readMtxFile( std::istream& file,
                                   Matrix& matrix,
                                   bool verbose )
{
   typedef typename Matrix::IndexType IndexType;
   typedef typename Matrix::RealType RealType;

   tnlString line;
   bool dimensionsLine( false ), formatOk( false );
   tnlList< tnlString > parsedLine;
   IndexType numberOfElements( 0 );
   IndexType size( 0 );
   bool symmetric( false );
   tnlVector< IndexType, tnlHost, IndexType > rowLengths;
   if( verbose )
      cout << "Counting the non-zero elements in rows..." << endl;
   while( line.getLine( file ) )
   {
      if( ! formatOk )
      {
         formatOk = checkMtxHeader( line, symmetric );
         if( formatOk && verbose )
         {
          if( symmetric )
             cout << "The matrix is SYMMETRIC." << endl;
         }
         continue;
      }
      if( line[ 0 ] == '%' ) continue;
      if( ! formatOk )
      {
         cerr << "Uknown format of the file. We expect line like this:" << endl;
         cerr << "%%MatrixMarket matrix coordinate real general" << endl;
         return false;
      }

      if( ! dimensionsLine )
      {
         parsedLine. EraseAll();
         line. parse( parsedLine );
         if( parsedLine. getSize() != 3 )
         {
           cerr << "Wrong number of parameters in the matrix header." << endl;
           return false;
         }
         const IndexType rows = atoi( parsedLine[ 0 ]. getString() );
         const IndexType columns = atoi( parsedLine[ 1 ]. getString() );
         numberOfElements = atoi( parsedLine[ 1 ]. getString() );
         cout << "Matrix rows:       " << setw( 9 ) << right << rows << endl;
         cout << "Matrix columns:       " << setw( 9 ) << right << columns << endl;

         if( rows <= 0 || columns <= 0 )
         {
           cerr << "Wrong parameters in the matrix header." << endl;
           return false;
         }
         if( ! matrix.setDimensions( rows, columns ) ||
             ! rowLengths.setSize( rows ) )
         {
            cerr << "Not enough memory to allocate the sparse or the full matrix for testing." << endl;
            return false;
         }
         rowLengths.setValue( 0 );

         dimensionsLine = true;
         continue;
      }
      if( parsedLine. getSize() != 3 )
      {
         cerr << "Wrong number of parameters in the matrix row at line:" << line << endl;
         return false;
      }
      parsedLine. EraseAll();
      line. parse( parsedLine );
      const IndexType row = atoi( parsedLine[ 0 ]. getString() );
      const IndexType column = atoi( parsedLine[ 1 ]. getString() );
      numberOfElements ++;
      if( verbose )
         cout << "Parsed thousands of elements:   " << setw( 9 ) << right << numberOfElements / 1000 << "\r" << flush;
      rowLengths[ row ]++;
      if( symmetric && row != column )
         rowLengths[ column ]++;
   }
   if( ! matrix.setRowLengths( rowLengths ) )
   {
      cerr << "Not enough memory to allocate the matrix." << endl;
      return false;
   }

   /****
    * Read the matrix elements
    */
   if( verbose )
      cout << endl;
   formatOk = false;
   dimensionsLine = false;
   file.seekg( 0 );
   IndexType parsedElements( 0 );
   if( verbose )
      cout << "Reading the matrix elements ..." << endl;
   while( line.getLine( file ) )
   {
      if( ! formatOk )
      {
         formatOk = checkMtxHeader( line, symmetric );
         continue;
      }
      if( ! dimensionsLine )
      {
         dimensionsLine = true;
         continue;
      }
      parsedLine.EraseAll();
      line.parse( parsedLine );
      const IndexType row = atoi( parsedLine[ 0 ].getString() );
      const IndexType column = atoi( parsedLine[ 1 ].getString() );
      const RealType value = ( RealType ) atof( parsedLine[ 2 ].getString() );
      matrix.setElement( row, column, value );
      if( symmetric && row != column )
         matrix.setElement( column, row, value );
      parsedElements++;
      if( verbose )
         cout << parsedElements << " / " << numberOfElements << "                       \r " << flush;
   }
   return true;
}

inline bool tnlMatrixReader::checkMtxHeader( const tnlString& header,
                                             bool& symmetric )
{
   tnlList< tnlString > parsedLine;
   header.parse( parsedLine );
   if( parsedLine. getSize() < 5 )
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


#endif /* TNLMATRIXREADER_IMPL_H_ */
