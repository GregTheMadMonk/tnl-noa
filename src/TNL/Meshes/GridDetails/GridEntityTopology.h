/***************************************************************************
                          GridEntityTopology.h  -  description
                             -------------------
    begin                : Nov 13, 2015
    copyright            : (C) 2015 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

namespace TNL {
namespace Meshes {


template< typename Grid,
          int EntityDimension,
          typename EntityOrientation_,
          typename EntityProportions_ >
class GridEntityTopology
{
   public:
 
      typedef Grid GridType;
 
      static constexpr int meshDimension = GridType::getMeshDimension();
 
      static constexpr int entityDimension = EntityDimension;
 
      typedef EntityOrientation_ EntityOrientation;
 
      typedef EntityProportions_ EntityProportions;
 
      // TODO: restore when CUDA allows it
   //static_assert( meshDimension == EntityOrientation_::size,
   //               "Entity orientation is not a proper static multiindex." );
};

} // namespace Meshes
} // namespace TNL

