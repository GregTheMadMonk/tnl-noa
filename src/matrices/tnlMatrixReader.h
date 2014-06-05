/***************************************************************************
                          tnlMatrixReader.h  -  description
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

#ifndef TNLMATRIXREADER_H_
#define TNLMATRIXREADER_H_

#include <istream>
#include <core/tnlString.h>
#include <core/vectors/tnlVector.h>

template< typename Matrix >
class tnlMatrixReader
{
   public:

   typedef typename Matrix::IndexType IndexType;
   typedef typename Matrix::RealType RealType;

   static bool readMtxFile( const tnlString& fileName,
                            Matrix& matrix,
                            bool verbose = false );

   static bool readMtxFile( std::istream& file,
                            Matrix& matrix,
                            bool verbose = false );

   static bool verifyMtxFile( std::istream& file,
                              const Matrix& matrix,
                              bool verbose = false );

   static bool findLineByElement( std::istream& file,
                                  const IndexType& row,
                                  const IndexType& column,
                                  tnlString& line,
                                  IndexType& lineNumber );
   protected:

   static bool checkMtxHeader( const tnlString& header,
                               bool& symmetric );

   static bool readMtxHeader( std::istream& file,
                              IndexType& rows,
                              IndexType& columns,
                              bool& symmetricMatrix,
                              bool verbose );

   static bool computeRowLengthsFromMtxFile( std::istream& file,
                                             tnlVector< int, tnlHost, int >& rowLengths,
                                             bool symmetricMatrix,
                                             bool verbose );

   static bool readMatrixElementsFromMtxFile( std::istream& file,
                                              Matrix& matrix,
                                              bool symmetricMatrix,
                                              bool verbose );

   static bool parseMtxLineWithElement( const tnlString& line,
                                        IndexType& row,
                                        IndexType& column,
                                        RealType& value );

};


#include <implementation/matrices/tnlMatrixReader_impl.h>

#endif /* TNLMATRIXREADER_H_ */
