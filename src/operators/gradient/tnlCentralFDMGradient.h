/***************************************************************************
                          tnlCentralFDMGradient.h  -  description
                             -------------------
    begin                : Apr 26, 2013
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

#ifndef TNLCENTRALFDMGRADIENT_H_
#define TNLCENTRALFDMGRADIENT_H_

#include <mesh/tnlGrid.h>
#include <mesh/tnlIdenticalGridGeometry.h>
#include <core/tnlHost.h>

template< typename Mesh >
class tnlCentralFDMGradient
{
};

template< typename Real,
          typename Device,
          typename Index,
          template< int, typename, typename, typename > class GridGeometry >
class tnlCentralFDMGradient< tnlGrid< 2, Real, Device, Index, GridGeometry > >
{
   public:

   typedef Real RealType;
   typedef Device DeviceType;
   typedef Index IndexType;
   typedef tnlGrid< 2, Real, Device, Index, GridGeometry > MeshType;
   typedef typename MeshType :: CoordinatesType CoordinatesType;
   typedef typename MeshType :: VertexType VertexType;

   tnlCentralFDMGradient();

   void bindMesh( const MeshType& mesh );

   template< typename Vector >
   void setFunction( Vector& f ); // TODO: add const

   void getGradient( const Index& i,
                     VertexType& grad_f ) const;
   protected:

   // TODO: change to ConstSharedVector
   tnlSharedVector< RealType, DeviceType, IndexType > f;

   const MeshType* mesh;
};

template< typename Real,
          typename Device,
          typename Index >
class tnlCentralFDMGradient< tnlGrid< 2, Real, Device, Index, tnlIdenticalGridGeometry > >
{
   public:

   typedef Real RealType;
   typedef Device DeviceType;
   typedef Index IndexType;
   typedef tnlGrid< 2, Real, Device, Index, tnlIdenticalGridGeometry > MeshType;
   typedef typename MeshType :: CoordinatesType CoordinatesType;
   typedef typename MeshType :: VertexType VertexType;

   tnlCentralFDMGradient();

   void bindMesh( const MeshType& mesh );

   template< typename Vector >
   void setFunction( Vector& f ); // TODO: add const

   void getGradient( const Index& i,
                     VertexType& grad_f ) const;
   protected:

   // TODO: change to ConstSharedVector
   tnlSharedVector< RealType, DeviceType, IndexType > f;

   const MeshType* mesh;
};


#include <implementation/operators/gradient/tnlCentralFDMGradient_impl.h>

#endif
