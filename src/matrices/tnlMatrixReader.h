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

class tnlMatrixReader
{
   public:

   template< typename Matrix >
   static bool readMtxFile( std::istream& file,
                            Matrix& matrix );
};


#include <implementation/matrices/tnlMatrixReader_impl.h>

#endif /* TNLMATRIXREADER_H_ */
