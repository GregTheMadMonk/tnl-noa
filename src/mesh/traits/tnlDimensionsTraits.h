/***************************************************************************
                          tnlDimensionsTraits.h  -  description
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

#ifndef TNLDIMENSIONSTRAITS_H_
#define TNLDIMENSIONSTRAITS_H_

template< int Dimensions >
class tnlDimensionsTraits
{
   public:

   enum { value = Dimensions };

   typedef tnlDimensionsTraits< Dimensions - 1 > Previous;
};


#endif /* TNLDIMENSIONSTRAITS_H_ */
