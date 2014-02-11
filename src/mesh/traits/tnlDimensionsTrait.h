/***************************************************************************
                          tnlDimensionsTrait.h  -  description
                             -------------------
    begin                : Feb 11, 2014
    copyright            : (C) 2014 by Tomas Oberhuber
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

#ifndef TNLDIMENSIONSTRAIT_H_
#define TNLDIMENSIONSTRAIT_H_

template< int Dimensions >
class tnlDimensionsTrait
{
   public:

   enum { dimensions = Dimensions };

   typedef tnlDimensionsTrait< Dimensions - 1 > Previous;
};


#endif /* TNLDIMENSIONSTRAIT_H_ */
