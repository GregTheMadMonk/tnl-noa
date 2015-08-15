/***************************************************************************
                          tnlDimensionsTag.h  -  description
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

#ifndef TNLDIMENSIONSTAG_H_
#define TNLDIMENSIONSTAG_H_

#include <core/tnlAssert.h>

template< int Dimensions >
class tnlDimensionsTag
{
   public:

      static const int value = Dimensions;

      typedef tnlDimensionsTag< Dimensions - 1 > Decrement;

      tnlStaticAssert( value >= 0, "The value of the dimensions cannot be negative." );
};

template<>
class tnlDimensionsTag< 0 >
{
   public:
   
      static const int value = 0;
};

#endif /* TNLDIMENSIONSTAG_H_ */
