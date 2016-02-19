/***************************************************************************
                          tnlNeighbourGridEntityGetter.h  -  description
                             -------------------
    begin                : Nov 23, 2015
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

#ifndef TNLNEIGHBOURGRIDENTITYGETTER_H
#define	TNLNEIGHBOURGRIDENTITYGETTER_H

#include <core/tnlAssert.h>

enum tnlGridEntityStencilStorage
{ 
   tnlGridEntityNoStencil = 0,
   tnlGridEntityCrossStencil,
   tnlGridEntityFullStencil
};

template< int storage >
class tnlGridEntityStencilStorageTag
{
   public:
      
      static const int stencilStorage = storage;
};

template< typename GridEntity,
          int NeighbourEntityDimensions,
          typename EntityStencilTag = 
            tnlGridEntityStencilStorageTag< GridEntity::ConfigType::template neighbourEntityStorage< GridEntity >( NeighbourEntityDimensions ) > >
class tnlNeighbourGridEntityGetter
{
   public:

      // TODO: not all specializations are implemented yet
      
      __cuda_callable__
      tnlNeighbourGridEntityGetter( const GridEntity& entity )
      {
         //tnlAssert( false, );
      };
      
      __cuda_callable__
      void refresh( const typename GridEntity::GridType& grid,
                    const typename GridEntity::IndexType& entityIndex )
      {
         //tnlAssert( false, );
      };

};


#endif	/* TNLNEIGHBOURGRIDENTITYGETTER_H */

