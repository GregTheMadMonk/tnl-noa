/***************************************************************************
                          DistributedGrid_2D.h  -  description
                             -------------------
    begin                : January 09, 2018
    copyright            : (C) 2018 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/Meshes/Grid.h>
#include <TNL/Meshes/DistributedMeshes/DistributedGrid_Base.h>

namespace TNL {
namespace Meshes { 
namespace DistributedMeshes {


template< typename RealType, typename Device, typename Index >
class DistributedMesh< Grid< 2, RealType, Device, Index > >: public DistributedGrid_Base<2, RealType, Device, Index >
{
   public:

/*    using typename DistributedGrid_Base<2, RealType, Device, Index >::IndexType;
      using typename DistributedGrid_Base<2, RealType, Device, Index >::GridType;
      using typename DistributedGrid_Base<2, RealType, Device, Index >::PointType;
      using typename DistributedGrid_Base<2, RealType, Device, Index >::CoordinatesType;*/
      typedef typename DistributedGrid_Base<2, RealType, Device, Index >::CoordinatesType CoordinatesType;
      typedef typename DistributedGrid_Base<2, RealType, Device, Index >::IndexType IndexType;
      typedef typename DistributedGrid_Base<2, RealType, Device, Index >::GridType GridType;
      typedef typename DistributedGrid_Base<2, RealType, Device, Index >::PointType PointType;
      typedef typename DistributedGrid_Base<2, RealType, Device, Index >::SubdomainOverlapsType SubdomainOverlapsType;

      bool setup( const Config::ParameterContainer& parameters,
                  const String& prefix );
            
      void setupGrid( GridType& grid );
       
      String printProcessCoords() const;

      String printProcessDistr() const;
             
      void writeProlog( Logger& logger ) const;

};

} // namespace DistributedMeshes
} // namespace Meshes
} // namespace TNL

#include <TNL/Meshes/DistributedMeshes/DistributedGrid_2D.hpp>
