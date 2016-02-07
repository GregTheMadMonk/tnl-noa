/***************************************************************************
                          tnlPDEOperatorEocTestMeshSetter.h  -  description
                             -------------------
    begin                : Feb 1, 2016
    copyright            : (C) 2016 by Tomas Oberhuber
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

#ifndef TNLPDEOPERATOREOCTESTMESHSETTER_H
#define	TNLPDEOPERATOREOCTESTMESHSETTER_H

#include <mesh/tnlGrid.h>

template< typename Mesh >
class tnlPDEOperatorEocTestMeshSetter
{
};

template< typename Real,
          typename Device,
          typename Index >
class tnlPDEOperatorEocTestMeshSetter< tnlGrid< 1, Real, Device, Index > >
{
   public:
      
      typedef tnlGrid< 1, Real, Device, Index > MeshType;
      typedef typename MeshType::VertexType VertexType;
      typedef typename MeshType::CoordinatesType CoordinatesType;
      typedef Real RealType;
      typedef Device DeviceType;
      typedef Index IndexType;
      
      static bool setup( MeshType& mesh, const IndexType meshSize )
      {
         VertexType origin, proportions;
         origin.x() = -2.0;
         proportions.x() = 4.0;
         mesh.setDomain( origin, proportions );

         CoordinatesType dimensions;
         dimensions.x() = meshSize;
         mesh.setDimensions( dimensions );
      }
};

template< typename Real,
          typename Device,
          typename Index >
class tnlPDEOperatorEocTestMeshSetter< tnlGrid< 2, Real, Device, Index > >
{
   public:
      
      typedef tnlGrid< 2, Real, Device, Index > MeshType;
      typedef typename MeshType::VertexType VertexType;
      typedef typename MeshType::CoordinatesType CoordinatesType;
      typedef Real RealType;
      typedef Device DeviceType;
      typedef Index IndexType;
      
      static bool setup( MeshType& mesh, const IndexType meshSize )
      {
         VertexType origin, proportions;
         origin.x() = -1.0;
         origin.y() = -1.0;
         proportions.x() = 2.0;
         proportions.y() = 2.0;
         mesh.setDomain( origin, proportions );

         CoordinatesType dimensions;
         dimensions.x() = meshSize;
         dimensions.y() = meshSize;
         mesh.setDimensions( dimensions );         
      }
};

template< typename Real,
          typename Device,
          typename Index >
class tnlPDEOperatorEocTestMeshSetter< tnlGrid< 3, Real, Device, Index > >
{
   public:
      
      typedef tnlGrid< 3, Real, Device, Index > MeshType;
      typedef typename MeshType::VertexType VertexType;
      typedef typename MeshType::CoordinatesType CoordinatesType;
      typedef Real RealType;
      typedef Device DeviceType;
      typedef Index IndexType;

      static bool setup( MeshType& mesh, const IndexType meshSize )      
      {
         VertexType origin, proportions;
         origin.x() = -1.0;
         origin.y() = -1.0;
         origin.z() = -1.0;
         proportions.x() = 2.0;
         proportions.y() = 2.0;
         proportions.z() = 2.0;
         mesh.setDomain( origin, proportions );

         CoordinatesType dimensions;
         dimensions.x() = meshSize;
         dimensions.y() = meshSize;
         dimensions.z() = meshSize;
         mesh.setDimensions( dimensions );         
      }
};



#endif	/* TNLPDEOPERATOREOCTESTMESHSETTER_H */

