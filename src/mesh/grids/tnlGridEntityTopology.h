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

template< typename Grid,
          int EntityDimensions,
          typename EntityOrientation >
class tnlGridEntityTopology
{
   static_assert( false, "" );
};

template< int MeshDimensions,
          typename Real,
          typename Device,
          typename Index,
          int EntityDimensions,
          typename EntityOrientation_ >
class tnlGridEntityTopology< tnlGrid< MeshDimensions, Real, Device, Index >,
                             EntityDimensions,
                             EntityOrientation_ >
{
   static_assert( MeshDimensions == EntityOrientation_::size, "Entity orientation is not a proper static multiindex." );
   public:
      
      typedef tnlGrid< MeshDimensions, Real, Device, Index > Grid;
      
      static const int meshDimensions = MeshDimensions;
      
      static const int entityDimensions = EntityDimensions;
      
      typedef EntityOrientation_ EntityOrientation;
};



#endif	/* TNLGRIDTOPOLOGIES_H */

