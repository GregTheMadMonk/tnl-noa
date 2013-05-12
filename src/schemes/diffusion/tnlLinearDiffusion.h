/***************************************************************************
                          tnlLinearDiffusion.h  -  description
                             -------------------
    begin                : Apr 29, 2013
    copyright            : (C) 2013 by Tomas Oberhuber
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

#ifndef TNLLINEARDIFFUSION_H_
#define TNLLINEARDIFFUSION_H_

#include <mesh/tnlGrid.h>
#include <mesh/tnlIdenticalGridGeometry.h>
#include <core/tnlHost.h>
#include <core/tnlSharedVector.h>

template< typename Mesh >
class tnlLinearDiffusion
{
};

template< typename Real,
          typename Device,
          typename Index,
          template< int, typename, typename, typename > class GridGeometry >
class tnlLinearDiffusion< tnlGrid< 2, Real, Device, Index, GridGeometry > >
{
   public:

   typedef Real RealType;
   typedef Device DeviceType;
   typedef Index IndexType;
   typedef tnlGrid< 2, Real, Device, Index, GridGeometry > MeshType;
   typedef typename MeshType :: CoordinatesType CoordinatesType;
   typedef typename MeshType :: VertexType VertexType;

   tnlLinearDiffusion();

   void bindMesh( const MeshType& mesh );

   template< typename Vector >
   void setFunction( Vector& f ); // TODO: add const

   RealType getDiffusion( const Index& i ) const;
   protected:

   // TODO: change to ConstSharedVector
   tnlSharedVector< RealType, DeviceType, IndexType > f;

   const MeshType* mesh;
};

template< typename Real, typename Device, typename Index >
class tnlLinearDiffusion< tnlGrid< 2, Real, Device, Index, tnlIdenticalGridGeometry > >
{
   public:

   typedef Real RealType;
   typedef Device DeviceType;
   typedef Index IndexType;
   typedef typename tnlGrid< 2, Real, Device, Index, tnlIdenticalGridGeometry > :: CoordinatesType CoordinatesType;
   typedef typename tnlGrid< 2, Real, Device, Index, tnlIdenticalGridGeometry > :: VertexType VertexType;

   tnlLinearDiffusion();

   void bindMesh( const tnlGrid< 2, RealType, DeviceType, IndexType, tnlIdenticalGridGeometry >& mesh );

   template< typename Vector >
   void setFunction( Vector& f ); // TODO: add const

   RealType getDiffusion( const Index& i ) const;
   protected:

   // TODO: change to ConstSharedVector
   tnlSharedVector< RealType, DeviceType, IndexType > f;

   const tnlGrid< 2, RealType, DeviceType, IndexType, tnlIdenticalGridGeometry >* mesh;
};


#include <implementation/schemes/diffusion/tnlLinearDiffusion_impl.h>

#endif
