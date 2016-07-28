/***************************************************************************
                          tnlDomain.h  -  description
                             -------------------
    begin                : Nov 8, 2015
    copyright            : (C) 2015 by oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */


#pragma once

namespace TNL {
namespace Functions {   

enum tnlDomainType { NonspaceDomain, SpaceDomain, MeshDomain, MeshInteriorDomain, MeshBoundaryDomain };

template< int Dimensions,
          tnlDomainType DomainType = SpaceDomain >
class tnlDomain
{
   public:
 
      typedef void DeviceType;
 
      static const int dimensions = Dimensions;
      static constexpr int getDimensions() { return Dimensions; }
 
      static constexpr tnlDomainType getDomainType() { return DomainType; }
};

} // namespace Functions
} // namespace TNL

