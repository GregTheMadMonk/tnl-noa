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

#include <core/tnlAssert.h>

template< int Dimensions >
class tnlDimensionsTraits
{
   public:

   enum { value = Dimensions };

   typedef tnlDimensionsTraits< Dimensions - 1 > Previous;

   tnlStaticAssert( value >= 0, "The value of the dimensions cannot be negative." );
};

template<>
class tnlDimensionsTraits< 0 >
{
   public:
   enum { value = 0 };
};

#endif /* TNLDIMENSIONSTRAITS_H_ */
