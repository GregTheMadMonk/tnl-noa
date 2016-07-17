/***************************************************************************
                          tnlMatrixWriter.h  -  description
                             -------------------
    begin                : Dec 18, 2013
    copyright            : (C) 2013 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <ostream>

namespace TNL {

template< typename Matrix >
class tnlMatrixWriter
{
   public:

   typedef typename Matrix::IndexType IndexType;
   typedef typename Matrix::RealType RealType;

   static bool writeToGnuplot( std::ostream str,
                               const Matrix& matrix,
                               bool verbose = false );

   static bool writeToEps( std::ostream str,
                           const Matrix& matrix,
                           bool verbose = false );

   protected:

   static bool writeEpsHeader( std::ostream str,
                               const Matrix& matrix,
                               const int elementSize );

   static bool writeEpsBody( std::ostream str,
                             const Matrix& matrix,
                             const int elementSize );


};

} // namespace TNL
