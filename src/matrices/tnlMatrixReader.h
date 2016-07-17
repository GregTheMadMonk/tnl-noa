/***************************************************************************
                          tnlMatrixReader.h  -  description
                             -------------------
    begin                : Dec 14, 2013
    copyright            : (C) 2013 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <istream>
#include <core/tnlString.h>
#include <core/vectors/tnlVector.h>

namespace TNL {

template< typename Device >
class tnlMatrixReaderDeviceDependentCode
{};

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

   static bool readMtxFileHostMatrix( std::istream& file,
                                      Matrix& matrix,
                                      typename Matrix::CompressedRowsLengthsVector& rowLengths,
                                      bool verbose );


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

   static bool computeCompressedRowsLengthsFromMtxFile( std::istream& file,
                                             tnlVector< int, tnlHost, int >& rowLengths,
                                             const int columns,
                                             const int rows,
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

} // namespace TNL

#include <matrices/tnlMatrixReader_impl.h>
