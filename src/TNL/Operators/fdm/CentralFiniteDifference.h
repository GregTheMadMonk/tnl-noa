/***************************************************************************
                          CentralFiniteDifference.h  -  description
                             -------------------
    begin                : Jan 9, 2016
    copyright            : (C) 2016 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/Operators/fdm/FiniteDifferences.h>
#include <TNL/Operators/fdm/ExactDifference.h>
#include <TNL/Operators/Operator.h>

namespace TNL {
namespace Operators {   

template< typename Mesh,
          int Xdifference = 0,
          int YDifference = 0,
          int ZDifference = 0,
          typename RealType = typename Mesh::RealType,
          typename IndexType = typename Mesh::IndexType >
class CentralFiniteDifference
{
};

template< int Dimensions,
          typename MeshReal,
          typename MeshDevice,
          typename MeshIndex,
          int XDifference,
          int YDifference,
          int ZDifference,
          typename Real,
          typename Index >
class CentralFiniteDifference< tnlGrid< Dimensions, MeshReal, MeshDevice, MeshIndex >, XDifference, YDifference, ZDifference, Real, Index >
: public Operator< tnlGrid< Dimensions, MeshReal, MeshDevice, MeshIndex >,
                      Functions::MeshInteriorDomain, Dimensions, Dimensions, Real, Index >
{
   public:
 
      typedef tnlGrid< Dimensions, MeshReal, MeshDevice, MeshIndex > MeshType;
      typedef Real RealType;
      typedef MeshDevice DeviceType;
      typedef Index IndexType;
      typedef ExactDifference< Dimensions, XDifference, YDifference, ZDifference > ExactOperatorType;
 
      //static constexpr int getMeshDimensions() { return Dimensions; }
 
      static String getType()
      {
         return String( "CentralFiniteDifference< " ) +
            MeshType::getType() + ", " +
            String( XDifference ) + ", " +
            String( YDifference ) + ", " +
            String( ZDifference ) + ", " +
           TNL::getType< RealType >() + ", " +
           TNL::getType< IndexType >() + " >";
      }

 
      template< typename MeshFunction, typename MeshEntity >
      __cuda_callable__
      inline Real operator()( const MeshFunction& u,
                              const MeshEntity& entity,
                              const RealType& time = 0.0 ) const
      {
         static_assert( MeshFunction::getEntitiesDimensions() == Dimensions,
            "Finite differences can be evaluate only on mesh cells, i.e. the dimensions count of the mesh entities of mesh function must be the same as mesh dimensions count." );
         return FiniteDifferences< MeshType, Real, Index, XDifference, YDifference, ZDifference, 0, 0, 0 >::getValue( u, entity );
      };
};

} // namespace Operators
} // namespace TNL

