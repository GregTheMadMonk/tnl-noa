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

/**
 * \brief Helper class for reading of matrices from files.
 *
 * It supports [MTX format](https://math.nist.gov/MatrixMarket/formats.html).
 * Currently only [Coordinate Format](https://math.nist.gov/MatrixMarket/formats.html#coord) is supported.
 *
 * \tparam Matrix is a type of matrix into which we want to import the MTX file.
 */
template< typename Matrix,
          typename Device = typename Matrix::DeviceType >
class MatrixReader
{
   public:

      /**
       * \brief Type of matrix elements values.
       */
      using RealType = typename Matrix::RealType;

      /**
       * \brief Device where the matrix is allocated.
       */
      using DeviceType = typename Matrix::RealType;

      /**
       * \brief Type used for indexing of matrix elements.
       */
      using IndexType = typename Matrix::IndexType;

      /**
       * \brief Method for importing matrix from file with given filename.
       *
       * \param fileName is the name of the source file.
       * \param matrix is the target matrix.
       * \param verbose controls verbosity of the matrix import.
       *
       * \par Example
       * \include Matrices/MatrixWriterReaderExample.cpp
       * \par Output
       * \include Matrices/MatrixWriterReaderExample.out
       *
       */
      static void readMtx( const String& fileName,
                           Matrix& matrix,
                           bool verbose = false );

      /**
       * \brief Method for importing matrix from STL input stream.
       *
       * \param file is the input stream.
       * \param matrix is the target matrix.
       * \param verbose controls verbosity of the matrix import.
       */
      static void readMtx( std::istream& file,
                           Matrix& matrix,
                           bool verbose = false );

      using HostMatrix = typename Matrix::template Self< RealType, TNL::Devices::Host >;
};

/// This is to prevent from appearing in Doxygen documentation.
/// \cond HIDDEN_CLASS

template< typename Matrix >
class MatrixReader< Matrix, TNL::Devices::Host >
{
   public:

      /**
       * \brief Type of matrix elements values.
       */
      typedef typename Matrix::RealType RealType;

      /**
       * \brief Device where the matrix is allocated.
       */
      typedef typename Matrix::DeviceType DeviceType;

      /**
       * \brief Type used for indexing of matrix elements.
       */
      typedef typename Matrix::IndexType IndexType;

      /**
       * \brief Method for importing matrix from file with given filename.
       *
       * \param fileName is the name of the source file.
       * \param matrix is the target matrix.
       * \param verbose controls verbosity of the matrix import.
       *
       * \par Example
       * \include Matrices/MatrixWriterReaderExample.cpp
       * \par Output
       * \include Matrices/MatrixWriterReaderExample.out
       *
       */
      static void readMtx( const String& fileName,
                           Matrix& matrix,
                           bool verbose = false );

      /**
       * \brief Method for importing matrix from STL input stream.
       *
       * \param file is the input stream.
       * \param matrix is the target matrix.
       * \param verbose controls verbosity of the matrix import.
       */
      static void readMtx( std::istream& file,
                           Matrix& matrix,
                           bool verbose = false );

   protected:

      static void verifyMtxFile( std::istream& file,
                                 const Matrix& matrix,
                                 bool verbose = false );

      static bool findLineByElement( std::istream& file,
                                    const IndexType& row,
                                    const IndexType& column,
                                    String& line,
                                    IndexType& lineNumber );


      static void checkMtxHeader( const String& header,
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
                                                bool symmetricSourceMatrix,
                                                bool symmetricTargetMatrix,
                                                bool verbose );

      static void readMatrixElementsFromMtxFile( std::istream& file,
                                                 Matrix& matrix,
                                                 bool symmetricMatrix,
                                                 bool verbose );

      static void parseMtxLineWithElement( const String& line,
                                           IndexType& row,
                                           IndexType& column,
                                           RealType& value );
};
/// \endcond


} // namespace Matrices
} // namespace TNL

#include <TNL/Matrices/MatrixReader.hpp>
