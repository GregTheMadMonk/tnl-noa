/***************************************************************************
                          tnlFunction.h  -  description
                             -------------------
    begin                : Nov 8, 2015
    copyright            : (C) 2015 by oberhuber
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


#ifndef TNLFUNCTION_H
#define	TNLFUNCTION_H

#include <core/vectors/tnlStaticVector.h>

enum tnlDomainType { NonspaceDomain, SpaceDomain, MeshDomain, MeshInteriorDomain, MeshBoundaryDomain };
   /*MeshFunction,
                       SpaceDomain,
                       SpaceDomain };*/

template< int Dimensions,
          tnlDomainType DomainType = SpaceDomain >
class tnlDomain
{
   public:
      
      typedef void DeviceType;
      
      static const int dimensions = Dimensions;
      static constexpr int getDimensions() { return Dimensions; }
      
      static constexpr tnlDomainType getFunctionType() { return DomainType; }
};

#endif	/* TNLFUNCTION_H */

