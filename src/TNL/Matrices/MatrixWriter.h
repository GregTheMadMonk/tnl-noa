/***************************************************************************
                          MatrixWriter.h  -  description
                             -------------------
    begin                : Dec 18, 2013
    copyright            : (C) 2013 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <ostream>
#include <iostream>
#include <TNL/String.h>

namespace TNL {
namespace Matrices {

template< typename Matrix, typename Device = typename Matrix::DeviceType >
class MatrixWriter
{
   public:

      using RealType = typename Matrix::RealType;
      using DeviceType = typename Matrix::RealType;
      using IndexType = typename Matrix::IndexType;
      using HostMatrix = typename Matrix::Self< RealType, TNL::Devices::Host >;


      static void writeGnuplot( const TNL::String& fileName,
                                const Matrix& matrix,
                                bool verbose = false );


      static void writeGnuplot( std::ostream& str,
                                const Matrix& matrix,
                                bool verbose = false );

      static void writeEps( const TNL::String& fileName,
                            const Matrix& matrix,
                            bool verbose = false );

      static void writeEps( std::ostream& str,
                            const Matrix& matrix,
                            bool verbose = false );

      static void writeMtx( const TNL::String& fileName,
                            const Matrix& matrix,
                            bool verbose = false );

      static void writeMtx( std::ostream& str,
                            const Matrix& matrix,
                            bool verbose = false );
};

template< typename Matrix >
class MatrixWriter< Matrix, TNL::Devices::Host >
{
   public:

   typedef typename Matrix::IndexType IndexType;
   typedef typename Matrix::RealType RealType;

   static void writeGnuplot( const TNL::String& fileName,
                             const Matrix& matrix,
                             bool verbose = false );


   static void writeGnuplot( std::ostream& str,
                             const Matrix& matrix,
                             bool verbose = false );

   static void writeEps( const TNL::String& fileName,
                         const Matrix& matrix,
                         bool verbose = false );

   static void writeEps( std::ostream& str,
                         const Matrix& matrix,
                         bool verbose = false );

   static void writeMtx( const TNL::String& fileName,
                         const Matrix& matrix,
                         bool verbose = false );

   static void writeMtx( std::ostream& str,
                         const Matrix& matrix,
                         bool verbose = false );

   protected:

   static void writeEpsHeader( std::ostream& str,
                               const Matrix& matrix,
                               const int elementSize );

   static void writeEpsBody( std::ostream& str,
                             const Matrix& matrix,
                             const int elementSize,
                             bool verbose );
};



} // namespace Matrices
} // namespace TNL

#include <TNL/Matrices/MatrixWriter.hpp>
