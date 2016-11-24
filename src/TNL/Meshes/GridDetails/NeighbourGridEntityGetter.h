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
#include <TNL/Devices/Cuda.h>
#include <TNL/Meshes/GridEntityConfig.h>

namespace TNL {
namespace Meshes {

template< typename GridEntity,
          int NeighbourEntityDimension,
          typename EntityStencilTag =
            GridEntityStencilStorageTag< GridEntity::ConfigType::template neighbourEntityStorage< GridEntity >( NeighbourEntityDimension ) > >
class NeighbourGridEntityGetter
{
   public:

      // TODO: not all specializations are implemented yet
 
      __cuda_callable__
      NeighbourGridEntityGetter( const GridEntity& entity )
      {
         //TNL_ASSERT( false, );
      }
 
      __cuda_callable__
      void refresh( const typename GridEntity::GridType& grid,
                    const typename GridEntity::IndexType& entityIndex )
      {
         //TNL_ASSERT( false, );
      }

};

} // namespace Meshes
} // namespace TNL

