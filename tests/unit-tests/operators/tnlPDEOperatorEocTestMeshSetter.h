/***************************************************************************
                          tnlPDEOperatorEocTestMeshSetter.h  -  description
                             -------------------
    begin                : Feb 1, 2016
    copyright            : (C) 2016 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

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

         return true;
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

         return true;
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
 
         return true;
      }
};



#endif	/* TNLPDEOPERATOREOCTESTMESHSETTER_H */

