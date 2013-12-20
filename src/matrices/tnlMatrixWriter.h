/***************************************************************************
                          tnlMatrixWriter.h  -  description
                             -------------------
    begin                : Dec 18, 2013
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

#ifndef TNLMATRIXWRITER_H_
#define TNLMATRIXWRITER_H_

#include <ostream>

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


#endif /* TNLMATRIXWRITER_H_ */
