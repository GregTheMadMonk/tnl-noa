/***************************************************************************
                          BufferEntittiesHelper.h  -  description
                             -------------------
    begin                : March 1, 2018
    copyright            : (C) 2018 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/Devices/Host.h>
#include <TNL/Devices/Cuda.h>
#include <TNL/ParallelFor.h>

namespace TNL {
namespace Meshes {
namespace DistributedMeshes {


template < typename MeshFunctionType,
           typename PeriodicBoundariesMaskPointer,
           int dim,
           typename RealType=typename MeshFunctionType::MeshType::RealType,
           typename Device=typename MeshFunctionType::MeshType::DeviceType,
           typename Index=typename MeshFunctionType::MeshType::GlobalIndexType >
class BufferEntitiesHelper
{
};


template < typename MeshFunctionType,
           typename MaskPointer,
           typename RealType,
           typename Device,
           typename Index >
class BufferEntitiesHelper< MeshFunctionType, MaskPointer, 1, RealType, Device, Index >
{
   public:
      static void BufferEntities(
         MeshFunctionType& meshFunction,
         const MaskPointer& maskPointer,
         RealType* buffer,
         bool isBoundary,
         const Index& beginx,
         const Index& sizex,
         bool tobuffer )
      {
         auto mesh = meshFunction.getMesh();
         RealType* meshFunctionData = meshFunction.getData().getData();
         const typename MaskPointer::ObjectType* mask( nullptr );
         if( maskPointer )
            mask = &maskPointer.template getData< Device >();
         auto kernel = [tobuffer, mesh, buffer, isBoundary, meshFunctionData, mask, beginx ] __cuda_callable__ ( Index j )
         {
            typename MeshFunctionType::MeshType::Cell entity(mesh);
            entity.getCoordinates().x()=beginx+j;
            entity.refresh();
            if( ! isBoundary || ! mask || ( *mask )[ entity.getIndex() ] )
            {
               if( tobuffer )
                  buffer[ j ] = meshFunctionData[ entity.getIndex() ];
               else
                  meshFunctionData[ entity.getIndex() ] = buffer[ j ];
            }
         };
         ParallelFor< Device >::exec( 0, sizex, kernel );
      };
};


template< typename MeshFunctionType,
          typename MaskPointer,
          typename RealType,
          typename Device,
          typename Index  >
class BufferEntitiesHelper< MeshFunctionType, MaskPointer, 2, RealType, Device, Index >
{
   public:
      static void BufferEntities(
         MeshFunctionType& meshFunction,
         const MaskPointer& maskPointer,
         RealType* buffer,
         bool isBoundary,
         const Index& beginx,
         const Index& beginy,
         const Index& sizex,
         const Index& sizey,
         bool tobuffer)
      {
         auto mesh=meshFunction.getMesh();
         RealType* meshFunctionData = meshFunction.getData().getData();
         const typename MaskPointer::ObjectType* mask( nullptr );
         if( maskPointer )
            mask = &maskPointer.template getData< Device >();

         auto kernel = [ tobuffer, mask, mesh, buffer, isBoundary, meshFunctionData, beginx, sizex, beginy] __cuda_callable__ ( Index i, Index j )
         {
            typename MeshFunctionType::MeshType::Cell entity(mesh);
            entity.getCoordinates().x() = beginx + i;
            entity.getCoordinates().y() = beginy + j;
            entity.refresh();
            if( ! isBoundary || ! mask || ( *mask )[ entity.getIndex() ] )
            {
               if( tobuffer )
                  buffer[ j * sizex + i ] = meshFunctionData[ entity.getIndex() ];
               else
                  meshFunctionData[ entity.getIndex() ] = buffer[ j * sizex + i ];
            }
         };
         ParallelFor2D< Device >::exec( 0, 0, sizex, sizey, kernel );
      };
};


template< typename MeshFunctionType,
          typename MaskPointer,
          typename RealType,
          typename Device,
          typename Index >
class BufferEntitiesHelper< MeshFunctionType, MaskPointer, 3, RealType, Device, Index >
{
   public:
      static void BufferEntities(
         MeshFunctionType& meshFunction,
         const MaskPointer& maskPointer,
         RealType* buffer,
         bool isBoundary,
         const Index& beginx,
         const Index& beginy,
         const Index& beginz,
         const Index& sizex,
         const Index& sizey,
         const Index& sizez,
         bool tobuffer)
      {
         auto mesh=meshFunction.getMesh();
         RealType * meshFunctionData=meshFunction.getData().getData();
         const typename MaskPointer::ObjectType* mask( nullptr );
         if( maskPointer )
            mask = &maskPointer.template getData< Device >();
         auto kernel = [ tobuffer, mesh, mask, buffer, isBoundary, meshFunctionData, beginx, sizex, beginy, sizey, beginz] __cuda_callable__ ( Index i, Index j, Index k )
         {
            typename MeshFunctionType::MeshType::Cell entity(mesh);
            entity.getCoordinates().x() = beginx + i;
            entity.getCoordinates().y() = beginy + j;
            entity.getCoordinates().z() = beginz + k;
            entity.refresh();
            if( ! isBoundary || ! mask || ( *mask )[ entity.getIndex() ] )
            {
               if( tobuffer )
                  buffer[ k * sizex * sizey + j * sizex + i ] = meshFunctionData[ entity.getIndex() ];
               else
                  meshFunctionData[ entity.getIndex() ] = buffer[ k * sizex * sizey + j * sizex + i ];
            }
         };
         ParallelFor3D< Device >::exec( 0, 0, 0, sizex, sizey, sizez, kernel );
      };
};


} // namespace DistributedMeshes
} // namespace Meshes
} // namespace TNL
