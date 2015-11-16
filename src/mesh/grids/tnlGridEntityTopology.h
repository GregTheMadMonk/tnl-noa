/***************************************************************************
                          tnlGridEntityTopology.h  -  description
                             -------------------
    begin                : Nov 13, 2015
    copyright            : (C) 2015 by Tomas Oberhuber
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

#ifndef TNLGRIDTOPOLOGIES_H
#define	TNLGRIDTOPOLOGIES_H

/****
 * TODO: zavest entity proportions a pokusit se "vse" odvodit z toho 
 */

template< typename Grid,
          int EntityDimensions,
          typename EntityOrientation_,
          typename EntityProportions_ >
class tnlGridEntityTopology
{
   public:
      
      typedef Grid GridType;
      
      static const int meshDimensions = GridType::Dimensions;
      
      static const int entityDimensions = EntityDimensions;
      
      typedef EntityOrientation_ EntityOrientation;
           
      typedef EntityProportions_ EntityProportions;
      
   static_assert( meshDimensions == EntityOrientation_::size, "Entity orientation is not a proper static multiindex." );
};



#endif	/* TNLGRIDTOPOLOGIES_H */

