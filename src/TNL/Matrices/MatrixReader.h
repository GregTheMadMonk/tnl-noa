/***************************************************************************
                          MatrixReader.h  -  description
                             -------------------
    begin                : Dec 14, 2013
    copyright            : (C) 2013 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <istream>
#include <TNL/String.h>
#include <TNL/Containers/Vector.h>

namespace TNL {
namespace Matrices {

template< typename Device >
class MatrixReaderDeviceDependentCode
{};

template< typename Matrix >
class MatrixReader
{
   public:

   typedef typename Matrix::IndexType IndexType;
   typedef typename Matrix::DeviceType DeviceType;
   typedef typename Matrix::RealType RealType;

   static void readMtxFile( const String& fileName,
                            Matrix& matrix,
                            bool verbose = false,
                            bool symReader = false );

   static void readMtxFile( std::istream& file,
                            Matrix& matrix,
                            bool verbose = false,
                            bool symReader = false );

   static void readMtxFileHostMatrix( std::istream& file,
                                      Matrix& matrix,
                                      typename Matrix::CompressedRowLengthsVector& rowLengths,
                                      bool verbose,
                                      bool symReader );


   static void verifyMtxFile( std::istream& file,
                              const Matrix& matrix,
                              bool verbose = false );

   static bool findLineByElement( std::istream& file,
                                  const IndexType& row,
                                  const IndexType& column,
                                  String& line,
                                  IndexType& lineNumber );
   protected:

   static bool checkMtxHeader( const String& header,
                               bool& symmetric );

   static void readMtxHeader( std::istream& file,
                              IndexType& rows,
                              IndexType& columns,
                              bool& symmetricMatrix,
                              bool verbose );

   static void computeCompressedRowLengthsFromMtxFile( std::istream& file,
                                             Containers::Vector< int, DeviceType, int >& rowLengths,
                                             const int columns,
                                             const int rows,
                                             bool symmetricMatrix,
                                             bool verbose,
                                             bool symReader = false );

   static void readMatrixElementsFromMtxFile( std::istream& file,
                                              Matrix& matrix,
                                              bool symmetricMatrix,
                                              bool verbose,
                                              bool symReader );

   static void parseMtxLineWithElement( const String& line,
                                        IndexType& row,
                                        IndexType& column,
                                        RealType& value );
};

} // namespace Matrices
} // namespace TNL

#include <TNL/Matrices/MatrixReader_impl.h>
