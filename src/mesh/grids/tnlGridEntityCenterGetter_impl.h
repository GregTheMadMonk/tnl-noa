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
          typename Index >
class tnlGridEntityCenterGetter< tnlGrid< 1, Real, Device, Index >,
                                 typename tnlGrid< 1, Real, Device, Index >::Cell >
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
         tnlAssert( cellCoordinates.x() >= 0 && cellCoordinates.x() < this->getDimensions().x(),
         cerr << "cellCoordinates.x() = " << cellCoordinates.x()
              << " this->getDimensions().x() = " << this->getDimensions().x() );
         return this->origin.x() + ( cellCoordinates.x() + 0.5 ) * this->cellProportions.x();
      }
};

template< typename Real,
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
         tnlAssert( vertexCoordinates.x() >= 0 && vertexCoordinates.x() < this->getDimensions().x() + 1,
         cerr << "vertexCoordinates.x() = " << vertexCoordinates.x()
              << " this->getDimensions().x() = " << this->getDimensions().x() );
         return this->origin.x() + vertexCoordinates.x() * this->cellProportions.x();
      }
};

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
         tnlAssert( cellCoordinates.x() >= 0 && cellCoordinates.x() < this->getDimensions().x(),
              cerr << "cellCoordinates.x() = " << cellCoordinates.x()
                   << " this->getDimensions().x() = " << this->getDimensions().x() );
         tnlAssert( cellCoordinates.y() >= 0 && cellCoordinates.y() < this->getDimensions().y(),
              cerr << "cellCoordinates.y() = " << cellCoordinates.y()
                   << " this->getDimensions().y() = " << this->getDimensions().y() );

         return this->origin.x() + ( cellCoordinates.x() + 0.5 ) * this->cellProportions.x(),
                this->origin.y() + ( cellCoordinates.y() + 0.5 ) * this->cellProportions.y();

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
}

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
         tnlAssert( faceCoordinates.x() >= 0 && faceCoordinates.x() < this->getDimensions().x() + 1,
                    cerr << "faceCoordinates.x() = " << faceCoordinates.x()
                         << " this->getDimensions().x() + 1 = " << this->getDimensions().x() + 1 );
         tnlAssert( faceCoordinates.y() >= 0 && faceCoordinates.y() < this->getDimensions().y(),
                    cerr << "faceCoordinates.y() = " << faceCoordinates.y()
                         << " this->getDimensions().y() = " << this->getDimensions().y() );
         return this->origin.x() + faceCoordinates.x() * this->cellProportions.x(),
                this->origin.y() + ( faceCoordinates.y() + 0.5 ) * this->cellProportions.y();
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
         tnlAssert( faceCoordinates.x() >= 0 && faceCoordinates.x() < this->getDimensions().x(),
                    cerr << "faceCoordinates.x() = " << faceCoordinates.x()
                         << " this->getDimensions().x() = " << this->getDimensions().x() );
         tnlAssert( faceCoordinates.y() >= 0 && faceCoordinates.y() < this->getDimensions().y() + 1,
                    cerr << "faceCoordinates.y() = " << faceCoordinates.y()
                         << " this->getDimensions().y() + 1 = " << this->getDimensions().y() + 1 );
         return this->origin.x() + ( faceCoordinates.x() + 0.5 ) * this->cellProportions.x(),
                this->origin.y() + faceCoordinates.y() * this->cellProportions.y();
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
         tnlAssert( vertexCoordinates.x() >= 0 && vertexCoordinates.x() < this->getDimensions().x() + 1,
              cerr << "vertexCoordinates.x() = " << vertexCoordinates.x()
                   << " this->getDimensions().x() = " << this->getDimensions().x() );
         tnlAssert( vertexCoordinates.y() >= 0 && vertexCoordinates.y() < this->getDimensions().y() + 1,
              cerr << "vertexCoordinates.y() = " << vertexCoordinates.y()
                   << " this->getDimensions().y() = " << this->getDimensions().y() );

         return Vertex( this->origin.x() + vertexCoordinates.x() * this->cellProportions.x(),
                        this->origin.y() + vertexCoordinates.y() * this->cellProportions.y() );
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
         tnlAssert( cellCoordinates.x() >= 0 && cellCoordinates.x() < this->getDimensions().x(),
                    cerr << "cellCoordinates.x() = " << cellCoordinates.x()
                         << " this->getDimensions().x() = " << this->getDimensions().x() );
         tnlAssert( cellCoordinates.y() >= 0 && cellCoordinates.y() < this->getDimensions().y(),
                    cerr << "cellCoordinates.y() = " << cellCoordinates.y()
                         << " this->getDimensions().y() = " << this->getDimensions().y() );
         tnlAssert( cellCoordinates.z() >= 0 && cellCoordinates.z() < this->getDimensions().z(),
                    cerr << "cellCoordinates.z() = " << cellCoordinates.z()
                         << " this->getDimensions().z() = " << this->getDimensions().z() );


         return Vertex( this->origin.x() + ( cellCoordinates.x() + 0.5 ) * this->cellProportions.x(),
                        this->origin.y() + ( cellCoordinates.y() + 0.5 ) * this->cellProportions.y(),
                        this->origin.z() + ( cellCoordinates.z() + 0.5 ) * this->cellProportions.z() );
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
         tnlAssert( faceCoordinates.x() >= 0 && faceCoordinates.x() < this->getDimensions().x() + 1,
                    cerr << "faceCoordinates.x() = " << faceCoordinates.x()
                         << " this->getDimensions().x() + 1 = " << this->getDimensions().x() + 1 );
         tnlAssert( faceCoordinates.y() >= 0 && faceCoordinates.y() < this->getDimensions().y(),
                    cerr << "faceCoordinates.y() = " << faceCoordinates.y()
                         << " this->getDimensions().y() = " << this->getDimensions().y() );
         tnlAssert( faceCoordinates.z() >= 0 && faceCoordinates.z() < this->getDimensions().z(),
                    cerr << "faceCoordinates.z() = " << faceCoordinates.z()
                         << " this->getDimensions().z() = " << this->getDimensions().z() );
         return Vertex( this->origin.x() + faceCoordinates.x() * this->cellProportions().x(),
                        this->origin.y() + ( faceCoordinates.y() + 0.5 ) * this->cellProportions().y(),
                        this->origin.z() + ( faceCoordinates.y() + 0.5 ) * this->cellProportions().z() );         
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
         tnlAssert( faceCoordinates.x() >= 0 && faceCoordinates.x() < this->getDimensions().x(),
                    cerr << "faceCoordinates.x() = " << faceCoordinates.x()
                         << " this->getDimensions().x() = " << this->getDimensions().x() );
         tnlAssert( faceCoordinates.y() >= 0 && faceCoordinates.y() < this->getDimensions().y() + 1,
                    cerr << "faceCoordinates.y() = " << faceCoordinates.y()
                         << " this->getDimensions().y() + 1 = " << this->getDimensions().y() + 1 );
         tnlAssert( faceCoordinates.z() >= 0 && faceCoordinates.z() < this->getDimensions().z(),
                    cerr << "faceCoordinates.z() = " << faceCoordinates.z()
                         << " this->getDimensions().z() = " << this->getDimensions().z() );

         return Vertex( this->origin.x() + ( faceCoordinates.x() + 0.5 ) * this->cellProportions().x(),
                        this->origin.y() + faceCoordinates.y() * this->cellProportions().y(),
                        this->origin.z() + ( faceCoordinates.z() + 0.5 ) * this->cellProportions().z() );         
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
         tnlAssert( faceCoordinates.x() >= 0 && faceCoordinates.x() < this->getDimensions().x(),
                    cerr << "faceCoordinates.x() = " << faceCoordinates.x()
                         << " this->getDimensions().x() = " << this->getDimensions().x() );
         tnlAssert( faceCoordinates.y() >= 0 && faceCoordinates.y() < this->getDimensions().y(),
                    cerr << "faceCoordinates.y() = " << faceCoordinates.y()
                         << " this->getDimensions().y()= " << this->getDimensions().y() );
         tnlAssert( faceCoordinates.z() >= 0 && faceCoordinates.z() < this->getDimensions().z() + 1,
                    cerr << "faceCoordinates.z() = " << faceCoordinates.z()
                         << " this->getDimensions().z() + 1 = " << this->getDimensions().z() + 1 );
         return Vertex( this->origin.x() + ( faceCoordinates.x() + 0.5 ) * this->cellProportions().x(),
                        this->origin.y() + ( faceCoordinates.y() + 0.5 ) * this->cellProportions().y(),
                        this->origin.z() + faceCoordinates.z() * this->cellProportions().z() );         
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
         tnlAssert( edgeCoordinates.x() >= 0 && edgeCoordinates.x() < this->getDimensions().x(),
                    cerr << "edgeCoordinates.x() = " << edgeCoordinates.x()
                         << " this->getDimensions().x() = " << this->getDimensions().x() );
         tnlAssert( edgeCoordinates.y() >= 0 && edgeCoordinates.y() < this->getDimensions().y() + 1,
                    cerr << "edgeCoordinates.y() = " << edgeCoordinates.y()
                         << " this->getDimensions().y() + 1 = " << this->getDimensions().y() + 1 );
         tnlAssert( edgeCoordinates.z() >= 0 && edgeCoordinates.z() < this->getDimensions().z() + 1,
                    cerr << "edgeCoordinates.z() = " << edgeCoordinates.z()
                         << " this->getDimensions().z() + 1 = " << this->getDimensions().z() + 1 );
         return Vertex( this->origin.x() + ( edgeCoordinates.x() + 0.5 ) * this->cellProportions().x(),
                        this->origin.y() + edgeCoordinates.y() * this->cellProportions().y(),
                        this->origin.z() + edgeCoordinates.z() * this->cellProportions().z() );         
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
         tnlAssert( edgeCoordinates.x() >= 0 && edgeCoordinates.x() < this->getDimensions().x() + 1,
                    cerr << "edgeCoordinates.x() = " << edgeCoordinates.x()
                         << " this->getDimensions().x() + 1 = " << this->getDimensions().x() + 1 );
         tnlAssert( edgeCoordinates.y() >= 0 && edgeCoordinates.y() < this->getDimensions().y(),
                    cerr << "edgeCoordinates.y() = " << edgeCoordinates.y()
                         << " this->getDimensions().y() = " << this->getDimensions().y() );
         tnlAssert( edgeCoordinates.z() >= 0 && edgeCoordinates.z() < this->getDimensions().z() + 1,
                    cerr << "edgeCoordinates.z() = " << edgeCoordinates.z()
                         << " this->getDimensions().z() + 1 = " << this->getDimensions().z() + 1 );
         return Vertex( this->origin.x() + edgeCoordinates.x() * this->cellProportions().x(),
                        this->origin.y() + ( edgeCoordinates.y() + 0.5 ) * this->cellProportions().y(),
                        this->origin.z() + edgeCoordinates.z() * this->cellProportions().z() );
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
         tnlAssert( edgeCoordinates.x() >= 0 && edgeCoordinates.x() < this->getDimensions().x() + 1,
                    cerr << "edgeCoordinates.x() = " << edgeCoordinates.x()
                         << " this->getDimensions().x() = " << this->getDimensions().x() );
         tnlAssert( edgeCoordinates.y() >= 0 && edgeCoordinates.y() < this->getDimensions().y() + 1,
                    cerr << "edgeCoordinates.y() = " << edgeCoordinates.y()
                         << " this->getDimensions().y() + 1 = " << this->getDimensions().y() + 1 );
         tnlAssert( edgeCoordinates.z() >= 0 && edgeCoordinates.z() < this->getDimensions().z(),
                    cerr << "edgeCoordinates.z() = " << edgeCoordinates.z()
                         << " this->getDimensions().z() = " << this->getDimensions().z() );
         return Vertex( this->origin.x() + edgeCoordinates.x() * this->cellProportions().x(),
                        this->origin.y() + edgeCoordinates.y() * this->cellProportions().y(),
                        this->origin.z() + ( edgeCoordinates.z() + 0.5 ) * this->cellProportions().z() );         
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
         tnlAssert( vertexCoordinates.x() >= 0 && vertexCoordinates.x() < this->getDimensions().x() + 1,
                    cerr << "vertexCoordinates.x() = " << vertexCoordinates.x()
                         << " this->getDimensions().x() = " << this->getDimensions().x() );
         tnlAssert( vertexCoordinates.y() >= 0 && vertexCoordinates.y() < this->getDimensions().y() + 1,
                    cerr << "vertexCoordinates.y() = " << vertexCoordinates.y()
                         << " this->getDimensions().y() = " << this->getDimensions().y() );
         tnlAssert( vertexCoordinates.z() >= 0 && vertexCoordinates.z() < this->getDimensions().z() + 1,
                    cerr << "vertexCoordinates.z() = " << vertexCoordinates.z()
                         << " this->getDimensions().z() = " << this->getDimensions().z() );

         return Vertex( this->origin.x() + vertexCoordinates.x() * this->cellProportions.x(),
                        this->origin.y() + vertexCoordinates.y() * this->cellProportions.y(),
                        this->origin.z() + vertexCoordinates.z() * this->cellProportions.z() );         
      }
};


















#endif	/* TNLGRIDENTITYCENTERGETTER_IMPL_H */

