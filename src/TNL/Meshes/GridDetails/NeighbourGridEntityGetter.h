/***************************************************************************
                          NeighbourGridEntityGetter.h  -  description
                             -------------------
    begin                : Nov 23, 2015
    copyright            : (C) 2015 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/Assert.h>

namespace TNL {
namespace Meshes {

/*enum GridEntityStencilStorage
{ 
   GridEntityNoStencil = 0,
   GridEntityCrossStencil,
   GridEntityFullStencil
};*/

template< int storage >
class GridEntityStencilStorageTag
{
   public:
 
      static const int stencilStorage = storage;
};

template< typename GridEntity,
          int NeighbourEntityDimensions,
          typename EntityStencilTag =
            GridEntityStencilStorageTag< GridEntity::ConfigType::template neighbourEntityStorage< GridEntity >( NeighbourEntityDimensions ) > >
class NeighbourGridEntityGetter
{
   public:

      // TODO: not all specializations are implemented yet
 
      __cuda_callable__
      NeighbourGridEntityGetter( const GridEntity& entity )
      {
         //Assert( false, );
      };
 
      __cuda_callable__
      void refresh( const typename GridEntity::GridType& grid,
                    const typename GridEntity::IndexType& entityIndex )
      {
         //Assert( false, );
      };

};

} // namespace Meshes
} // namespace TNL

