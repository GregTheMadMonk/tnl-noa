/***************************************************************************
                          tnlGridEntityCenterGetter_impl.h  -  description
                             -------------------
    begin                : Nov 15, 2015
    copyright            : (C) 2015 by Tomas Oberhuber
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

#ifndef TNLGRIDENTITYCENTERGETTER_IMPL_H
#define	TNLGRIDENTITYCENTERGETTER_IMPL_H

#include <mesh/grids/tnlGridEntityCenterGetter.h>
#include <mesh/grids/tnlGrid1D.h>
#include <mesh/grids/tnlGrid2D.h>
#include <mesh/grids/tnlGrid3D.h>

template< typename Real,
          typename Device,
          typename Index,
          typename EntityTopology >
class tnlGridEntityCenterGetter< tnlGrid< 1, Real, Device, Index >,
                                 EntityTopology >
{
   public:
      
      typedef tnlGrid< 1, Real, Device, Index > GridType;
      typedef typename GridType::CoordinatesType CoordinatesType;
      typedef typename GridType::VertexType VertexType;
      
      __cuda_callable__
      static VertexType getCenter( const CoordinatesType& cellCoordinates,
                                   const VertexType& origin,
                                   const VertexType& cellProportions )
      {
         return origin.x() + 
               ( cellCoordinates.x() + 0.5 * EntityTopology::EntityProportions::i1 ) * cellProportions.x();
      }
};

/*template< typename Real,
          typename Device,
          typename Index >
class tnlGridEntityCenterGetter< tnlGrid< 1, Real, Device, Index >,
                                 typename tnlGrid< 1, Real, Device, Index >::Vertex >
{
   public:
      
      typedef tnlGrid< 1, Real, Device, Index > GridType;
      typedef typename GridType::CoordinatesType CoordinatesType;
      typedef typename GridType::VertexType VertexType;
      
      __cuda_callable__
      static VertexType getCenter( const CoordinatesType& vertexCoordinates,
                                   const VertexType& origin,
                                   const VertexType& cellProportions )
      {
         return this->origin.x() + vertexCoordinates.x() * this->cellProportions.x();
      }
};*/

template< typename Real,
          typename Device,
          typename Index >
class tnlGridEntityCenterGetter< tnlGrid< 2, Real, Device, Index >,
                                 typename tnlGrid< 2, Real, Device, Index >::Cell >
{
   public:
      
      typedef tnlGrid< 2, Real, Device, Index > GridType;
      typedef typename GridType::CoordinatesType CoordinatesType;
      typedef typename GridType::VertexType VertexType;
      
      __cuda_callable__
      static VertexType getCenter( const CoordinatesType& cellCoordinates,
                                   const VertexType& origin,
                                   const VertexType& cellProportions )
      {
         return origin.x() + ( cellCoordinates.x() + 0.5 ) * cellProportions.x(),
                origin.y() + ( cellCoordinates.y() + 0.5 ) * cellProportions.y();

      }
};

template< typename Real,
          typename Device,
          typename Index,
          int nx,
          int ny >
class tnlGridEntityCenterGetter< tnlGrid< 2, Real, Device, Index >,
                                 typename tnlGrid< 2, Real, Device, Index >::template Face< nx, ny > >
{
   static_assert( false, "Wrong template parameters nx or ny for face normal. It can be one of (1,0) and (0,1)." );
};

template< typename Real,
          typename Device,
          typename Index >
class tnlGridEntityCenterGetter< tnlGrid< 2, Real, Device, Index >,
                                 typename tnlGrid< 2, Real, Device, Index >::template Face< 1, 0 > >
{
   public:
      
      typedef tnlGrid< 2, Real, Device, Index > GridType;
      typedef typename GridType::CoordinatesType CoordinatesType;
      typedef typename GridType::VertexType VertexType;
      
      __cuda_callable__
      static VertexType getCenter( const CoordinatesType& faceCoordinates,
                                   const VertexType& origin,
                                   const VertexType& cellProportions )
      {
         return origin.x() + faceCoordinates.x() * cellProportions.x(),
                origin.y() + ( faceCoordinates.y() + 0.5 ) * cellProportions.y();
      }
};

template< typename Real,
          typename Device,
          typename Index >
class tnlGridEntityCenterGetter< tnlGrid< 2, Real, Device, Index >,
                                 typename tnlGrid< 2, Real, Device, Index >::template Face< 0, 1 > >
{
   public:
      
      typedef tnlGrid< 2, Real, Device, Index > GridType;
      typedef typename GridType::CoordinatesType CoordinatesType;
      typedef typename GridType::VertexType VertexType;
      
      __cuda_callable__
      static VertexType getCenter( const CoordinatesType& faceCoordinates,
                                   const VertexType& origin,
                                   const VertexType& cellProportions )
      {
         return origin.x() + ( faceCoordinates.x() + 0.5 ) * cellProportions.x(),
                origin.y() + faceCoordinates.y() * cellProportions.y();
      }
};

template< typename Real,
          typename Device,
          typename Index >
class tnlGridEntityCenterGetter< tnlGrid< 2, Real, Device, Index >,
                                 typename tnlGrid< 2, Real, Device, Index >::Vertex >
{
   public:
      
      typedef tnlGrid< 2, Real, Device, Index > GridType;
      typedef typename GridType::CoordinatesType CoordinatesType;
      typedef typename GridType::VertexType VertexType;
      
      __cuda_callable__
      static VertexType getCenter( const CoordinatesType& vertexCoordinates,
                                   const VertexType& origin,
                                   const VertexType& cellProportions )
      {
         return origin.x() + vertexCoordinates.x() * cellProportions.x(),
                origin.y() + vertexCoordinates.y() * cellProportions.y();
      }
};

template< typename Real,
          typename Device,
          typename Index >
class tnlGridEntityCenterGetter< tnlGrid< 3, Real, Device, Index >,
                                 typename tnlGrid< 3, Real, Device, Index >::Cell >
{
   public:
      
      typedef tnlGrid< 3, Real, Device, Index > GridType;
      typedef typename GridType::CoordinatesType CoordinatesType;
      typedef typename GridType::VertexType VertexType;
      
      __cuda_callable__
      static VertexType getCenter( const CoordinatesType& cellCoordinates,
                                   const VertexType& origin,
                                   const VertexType& cellProportions )
      {
         return origin.x() + ( cellCoordinates.x() + 0.5 ) * cellProportions.x(),
                origin.y() + ( cellCoordinates.y() + 0.5 ) * cellProportions.y(),
                origin.z() + ( cellCoordinates.z() + 0.5 ) * cellProportions.z();
      }
};

template< typename Real,
          typename Device,
          typename Index,
          int nx,
          int ny,
          int nz >
class tnlGridEntityCenterGetter< tnlGrid< 3, Real, Device, Index >,
                                 typename tnlGrid< 3, Real, Device, Index >::template Face< nx, ny, nz > >
{
   static_assert( false, 
      "Wrong template parameters nx, ny or nz for a face normal. It can be one of (1,0,0), (0,1,0) and (0,0,1)." );
}

template< typename Real,
          typename Device,
          typename Index >
class tnlGridEntityCenterGetter< tnlGrid< 3, Real, Device, Index >,
                                 typename tnlGrid< 3, Real, Device, Index >::template Face< 1, 0, 0 > >
{
   public:
      
      typedef tnlGrid< 3, Real, Device, Index > GridType;
      typedef typename GridType::CoordinatesType CoordinatesType;
      typedef typename GridType::VertexType VertexType;
      
      __cuda_callable__
      static VertexType getCenter( const CoordinatesType& faceCoordinates,
                                   const VertexType& origin,
                                   const VertexType& cellProportions )
      {
         return origin.x() + faceCoordinates.x() * cellProportions().x(),
                origin.y() + ( faceCoordinates.y() + 0.5 ) * cellProportions().y(),
                origin.z() + ( faceCoordinates.y() + 0.5 ) * cellProportions().z();         
      }
};

template< typename Real,
          typename Device,
          typename Index >
class tnlGridEntityCenterGetter< tnlGrid< 3, Real, Device, Index >,
                                 typename tnlGrid< 3, Real, Device, Index >::template Face< 0, 1, 0 > >
{ 
   public:
      
      typedef tnlGrid< 3, Real, Device, Index > GridType;
      typedef typename GridType::CoordinatesType CoordinatesType;
      typedef typename GridType::VertexType VertexType;
      
      __cuda_callable__
      static VertexType getCenter( const CoordinatesType& faceCoordinates,
                                   const VertexType& origin,
                                   const VertexType& cellProportions )
      {
         return origin.x() + ( faceCoordinates.x() + 0.5 ) * cellProportions().x(),
                origin.y() + faceCoordinates.y() * cellProportions().y(),
                origin.z() + ( faceCoordinates.z() + 0.5 ) * cellProportions().z();         
      }
};

template< typename Real,
          typename Device,
          typename Index >
class tnlGridEntityCenterGetter< tnlGrid< 3, Real, Device, Index >,
                                 typename tnlGrid< 3, Real, Device, Index >::template Face< 0, 0, 1 > >
{ 
   public:
      
      typedef tnlGrid< 3, Real, Device, Index > GridType;
      typedef typename GridType::CoordinatesType CoordinatesType;
      typedef typename GridType::VertexType VertexType;
      
      __cuda_callable__
      static VertexType getCenter( const CoordinatesType& faceCoordinates,
                                   const VertexType& origin,
                                   const VertexType& cellProportions )
      {
         return origin.x() + ( faceCoordinates.x() + 0.5 ) * cellProportions().x(),
                origin.y() + ( faceCoordinates.y() + 0.5 ) * cellProportions().y(),
                origin.z() + faceCoordinates.z() * cellProportions().z();         
      }
};

template< typename Real,
          typename Device,
          typename Index,
          int dx,
          int dy,
          int dz >
class tnlGridEntityCenterGetter< tnlGrid< 3, Real, Device, Index >,
                                 typename tnlGrid< 3, Real, Device, Index >::template Face< dx, dy, dz > >
{
   static_assert( false, 
      "Wrong template parameters dx, dy or dz for an edge direction. It can be one of (1,0,0), (0,1,0) and (0,0,1)." );
}

template< typename Real,
          typename Device,
          typename Index >
class tnlGridEntityCenterGetter< tnlGrid< 3, Real, Device, Index >,
                                 typename tnlGrid< 3, Real, Device, Index >::template Edge< 1, 0, 0 > >
{
   public:
      
      typedef tnlGrid< 3, Real, Device, Index > GridType;
      typedef typename GridType::CoordinatesType CoordinatesType;
      typedef typename GridType::VertexType VertexType;
      
      __cuda_callable__
      static VertexType getCenter( const CoordinatesType& edgeCoordinates,
                                   const VertexType& origin,
                                   const VertexType& cellProportions )
      {
         return origin.x() + ( edgeCoordinates.x() + 0.5 ) * cellProportions().x(),
                origin.y() + edgeCoordinates.y() * cellProportions().y(),
                origin.z() + edgeCoordinates.z() * cellProportions().z();         
      }
};

template< typename Real,
          typename Device,
          typename Index >
class tnlGridEntityCenterGetter< tnlGrid< 3, Real, Device, Index >,
                                 typename tnlGrid< 3, Real, Device, Index >::template Edge< 0, 1, 0 > >
{ 
   public:
      
      typedef tnlGrid< 3, Real, Device, Index > GridType;
      typedef typename GridType::CoordinatesType CoordinatesType;
      typedef typename GridType::VertexType VertexType;
      
      __cuda_callable__
      static VertexType getCenter( const CoordinatesType& edgeCoordinates,
                                   const VertexType& origin,
                                   const VertexType& cellProportions )
      {
         return origin.x() + edgeCoordinates.x() * cellProportions().x(),
                origin.y() + ( edgeCoordinates.y() + 0.5 ) * cellProportions().y(),
                origin.z() + edgeCoordinates.z() * cellProportions().z();
      }
};

template< typename Real,
          typename Device,
          typename Index >
class tnlGridEntityCenterGetter< tnlGrid< 3, Real, Device, Index >,
                                 typename tnlGrid< 3, Real, Device, Index >::template Edge< 0, 0, 1 > >
{ 
   public:
      
      typedef tnlGrid< 3, Real, Device, Index > GridType;
      typedef typename GridType::CoordinatesType CoordinatesType;
      typedef typename GridType::VertexType VertexType;
      
      __cuda_callable__
      static VertexType getCenter( const CoordinatesType& edgeCoordinates,
                                   const VertexType& origin,
                                   const VertexType& cellProportions )
      {
         return origin.x() + edgeCoordinates.x() * cellProportions().x(),
                origin.y() + edgeCoordinates.y() * cellProportions().y(),
                origin.z() + ( edgeCoordinates.z() + 0.5 ) * cellProportions().z();         
      }
};

template< typename Real,
          typename Device,
          typename Index >
class tnlGridEntityCenterGetter< tnlGrid< 3, Real, Device, Index >,
                                 typename tnlGrid< 3, Real, Device, Index >::Vertex >
{
   public:
      
      typedef tnlGrid< 3, Real, Device, Index > GridType;
      typedef typename GridType::CoordinatesType CoordinatesType;
      typedef typename GridType::VertexType VertexType;
      
      __cuda_callable__
      static VertexType getCenter( const CoordinatesType& vertexCoordinates,
                                   const VertexType& origin,
                                   const VertexType& cellProportions )
      {
         return origin.x() + vertexCoordinates.x() * cellProportions.x(),
                origin.y() + vertexCoordinates.y() * cellProportions.y(),
                origin.z() + vertexCoordinates.z() * cellProportions.z();         
      }
};


#endif	/* TNLGRIDENTITYCENTERGETTER_IMPL_H */

