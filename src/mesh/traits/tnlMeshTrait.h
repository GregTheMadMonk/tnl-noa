/***************************************************************************
                          tnlMeshTrait.h  -  description
                             -------------------
    begin                : Feb 11, 2014
    copyright            : (C) 2014 by Tomas Oberhuber
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

#ifndef TNLMESHTRAIT_H_
#define TNLMESHTRAIT_H_

#include <core/vectors/tnlStaticVector.h>
#include <mesh/traits/tnlDimensionsTrait.h>
#include <mesh/tnlMeshEntity.h>

template< typename ConfigTag >
class tnlMeshTrait
{
   public:

   enum { meshDimensions = ConfigTag::CellTag::dimension };

   enum { worldDimensions = ConfigTag::dimWorld };

   typedef tnlDimensionsTrait< meshDimensions >                         DimensionsTrait;
   typedef tnlStaticVector< worldDimensions, typename ConfigTag::Real > PointType;
   typedef tnlMeshEntity< ConfigTag, typename ConfigTag::CellTag >      CellType;
};


#endif /* TNLMESHTRAIT_H_ */
