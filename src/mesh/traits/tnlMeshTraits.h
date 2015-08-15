/***************************************************************************
                          tnlMeshTraits.h  -  description
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

#ifndef TNLMESHTRAITS_H_
#define TNLMESHTRAITS_H_

#include <core/vectors/tnlStaticVector.h>
#include <mesh/tnlDimensionsTag.h>

template< typename ConfigTag,
          typename EntityTag >
class tnlMeshEntity;

template< typename ConfigTag >
class tnlMeshTraits
{
   public:

   enum { meshDimensions = ConfigTag::CellType::dimensions };

   enum { worldDimensions = ConfigTag::worldDimensions };

   typedef tnlDimensionsTag< meshDimensions >                            DimensionsTag;
   typedef tnlStaticVector< worldDimensions, typename ConfigTag::RealType > PointType;
   typedef tnlMeshEntity< ConfigTag, typename ConfigTag::CellType >          CellType;
};


#endif /* TNLMESHTRAITS_H_ */
