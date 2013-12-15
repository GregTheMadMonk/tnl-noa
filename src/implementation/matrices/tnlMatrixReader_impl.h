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

template< typename Matrix >
bool tnlMatrixReader::readMtxFile( std::istream& file,
                                   Matrix& matrix )
{
   typedef typename Matrix::IndexType IndexType;
   typedef typename Matrix::RealType RealType;

   tnlString line;
   bool dimensionsLine( false ), formatOk( false );
   tnlList< tnlString > parsed_line;
   Index numberOfElements( 0 );
   Index size( 0 );
   bool symmetric( false );
   tnlVector< IndexType, tnlHost, IndexType > rowLengths;
   while( line.getLine( file ) )
   {
      if( ! formatOk )
      {
         format_ok = checkMtxHeader( line, symmetric );
         if( format_ok && verbose )
         {
          if( symmetric )
             cout << "The matrix is SYMMETRIC." << endl;
         }
         continue;
      }
      if( line[ 0 ] == '%' ) continue;
      if( ! format_ok )
      {
         cerr << "Uknown format of the file. We expect line like this:" << endl;
         cerr << "%%MatrixMarket matrix coordinate real general" << endl;
         return false;
      }

      if( ! dimensionsLine )
      {
         parsed_line. EraseAll();
         line. parse( parsed_line );
         if( parsed_line. getSize() != 3 )
         {
           cerr << "Wrong number of parameters in the matrix header." << endl;
           return false;
         }
         const IndexType rows = atoi( parsed_line[ 0 ]. getString() );
         const IndexType columns = atoi( parsed_line[ 1 ]. getString() );
         cout << "Matrix rows:       " << setw( 9 ) << right << M << endl;
         cout << "Matrix columns:       " << setw( 9 ) << right << N << endl;

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
      if( parsed_line. getSize() != 3 )
      {
         cerr << "Wrong number of parameters in the matrix row at line:" << line << endl;
         return false;
      }
      parsed_line. EraseAll();
      line. parse( parsed_line );
      const IndexType row = atoi( parsed_line[ 0 ]. getString() );
      const IndexType column = atoi( parsed_line[ 1 ]. getString() );
      numberOfElements ++;
      if( verbose )
         cout << "Parsed elements:   " << setw( 9 ) << right << parsed_elements << "\r" << flush;
      rowLengths[ row ]++;
      if( symmetric && row != column )
         rowLength[ column ]++;
   }
   if( ! matrix.setRowLentghs( rowLengths ) )
   {
      cerr << "Not enough memory to allocate the matrix." << endl;
      return false;
   }
   /****
    * Read the matrix elements
    */
   formatOk = false;
   dimensionsLine = false;
   istream.seek( 0 );
   IndexType parsedElements( 0 );
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
      parsed_line.EraseAll();
      line.parse( parsed_line );
      const IndexType row = atoi( parsed_line[ 0 ].getString() );
      const IndexType column = atoi( parsed_line[ 1 ].getString() );
      const RealType value = ( Real ) atof( parsed_line[ 2 ].getString() );
      matrix.setElement( row, column, value );
      if( symmetric && row != column )
         matrix.setElement( column, row, value );
      parsedElements++;
      if( verbose )
         cout << parsedElements << " / " << totalElements << "                       \r " << flush;
   }
   return true;
}

#endif /* TNLMATRIXREADER_IMPL_H_ */
