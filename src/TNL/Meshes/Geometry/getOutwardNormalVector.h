/***************************************************************************
                          getOutwardNormalVector.h  -  description
                             -------------------
    begin                : Feb 14, 2020
    copyright            : (C) 2020 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/Meshes/Geometry/getEntityCenter.h>

namespace TNL {
namespace Meshes {

template< typename Grid, typename Config >
__cuda_callable__
typename Grid::PointType
getOutwardNormalVector( const Grid & grid,
                        const GridEntity< Grid, 0, Config > & face,
                        const typename Grid::PointType cellCenter )
{
   static_assert( Grid::getMeshDimension() == 1, "getOutwardNormalVector can be used only with faces." );
   const typename Grid::PointType faceCenter = getEntityCenter( grid, face );
   if( faceCenter.x() > cellCenter.x() )
      return {1};
   else
      return {-1};
}

template< typename Grid, typename Config >
__cuda_callable__
typename Grid::PointType
getOutwardNormalVector( const Grid & grid,
                        const GridEntity< Grid, 1, Config > & face,
                        const typename Grid::PointType cellCenter )
{
   static_assert( Grid::getMeshDimension() == 2, "getOutwardNormalVector can be used only with faces." );
   const typename Grid::PointType faceCenter = getEntityCenter( grid, face );
   if( face.getOrientation().x() != 0 ) {
      // x-normal face
      if( faceCenter.x() > cellCenter.x() )
         return {1, 0};
      else
         return {-1, 0};
   }
   else {
      // y-normal face
      if( faceCenter.y() > cellCenter.y() )
         return {0, 1};
      else
         return {0, -1};
   }
}

template< typename Grid, typename Config >
__cuda_callable__
typename Grid::PointType
getOutwardNormalVector( const Grid & grid,
                        const GridEntity< Grid, 2, Config > & face,
                        const typename Grid::PointType cellCenter )
{
   static_assert( Grid::getMeshDimension() == 3, "getOutwardNormalVector can be used only with faces." );
   const typename Grid::PointType faceCenter = getEntityCenter( grid, face );
   if( face.getOrientation().x() != 0 ) {
      // x-normal face
      if( faceCenter.x() > cellCenter.x() )
         return {1, 0, 0};
      else
         return {-1, 0, 0};
   }
   else if( face.getOrientation().y() != 0 ) {
      // y-normal face
      if( faceCenter.y() > cellCenter.y() )
         return {0, 1, 0};
      else
         return {0, -1, 0};
   }
   else  {
      // z-normal face
      if( faceCenter.z() > cellCenter.z() )
         return {0, 0, 1};
      else
         return {0, 0, -1};
   }
}

} // namespace Meshes
} // namespace TNL
