/***************************************************************************
                          Operator.h  -  description
                             -------------------
    begin                : Feb 10, 2016
    copyright            : (C) 2016 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/Functions/Domain.h>

namespace TNL {
namespace Operators {

template< typename Mesh,
          Functions::DomainType DomainType = Functions::MeshInteriorDomain,
          int PreimageEntitiesDimension = Mesh::getDimension(),
          int ImageEntitiesDimension = Mesh::getDimension(),
          typename Real = typename Mesh::RealType,
          typename Index = typename Mesh::IndexType >
class Operator : public Functions::Domain< Mesh::getDimension(), DomainType >
{
   public:
 
      typedef Mesh MeshType;
      typedef typename MeshType::RealType MeshRealType;
      typedef typename MeshType::DeviceType DeviceType;
      typedef typename MeshType::IndexType MeshIndexType;
      typedef Real RealType;
      typedef Index IndexType;
      typedef void ExactOperatorType;
 
      constexpr static int getDimension() { return MeshType::getDimension(); }
      constexpr static int getPreimageEntitiesDimension() { return PreimageEntitiesDimension; }
      constexpr static int getImageEntitiesDimension() { return ImageEntitiesDimension; }
 
      bool refresh( const RealType& time = 0.0 ) { return true; }
 
      bool deepRefresh( const RealType& time = 0.0 ) { return true; }
 
      template< typename MeshFunction >
      void setPreimageFunction( const MeshFunction& f ){}
};

} // namespace Operators
} // namespace TNL

