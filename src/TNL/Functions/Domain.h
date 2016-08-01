/***************************************************************************
                          Domain.h  -  description
                             -------------------
    begin                : Nov 8, 2015
    copyright            : (C) 2015 by oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */


#pragma once

namespace TNL {
namespace Functions {   

enum DomainType { NonspaceDomain, SpaceDomain, MeshDomain, MeshInteriorDomain, MeshBoundaryDomain };

template< int Dimensions,
          DomainType DomainType_ = SpaceDomain >
class Domain
{
   public:
 
      typedef void DeviceType;
 
      static const int dimensions = Dimensions;
      static constexpr int getDimensions() { return Dimensions; }
 
      static constexpr DomainType getDomainType() { return DomainType_; }
};

} // namespace Functions
} // namespace TNL

