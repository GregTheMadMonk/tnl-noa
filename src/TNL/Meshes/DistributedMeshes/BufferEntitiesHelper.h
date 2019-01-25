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
#include <TNL/Containers/StaticVector.h>

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
         const Containers::StaticVector<1,Index>& begin,
         const Containers::StaticVector<1,Index>& size,
         bool tobuffer )
      {

         Index beginx=begin.x();
         Index sizex=size.x();

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
         const Containers::StaticVector<2,Index>& begin,
         const Containers::StaticVector<2,Index>& size,
         bool tobuffer)
      {

         Index beginx=begin.x();
         Index beginy=begin.y();
         Index sizex=size.x();
         Index sizey=size.y();

         auto mesh=meshFunction.getMesh();
         RealType* meshFunctionData = meshFunction.getData().getData();      
         const typename MaskPointer::ObjectType* mask( nullptr );
         if( maskPointer )
            mask = &maskPointer.template getData< Device >();

         auto kernel = [ tobuffer, mask, mesh, buffer, isBoundary, meshFunctionData, beginx, sizex, beginy] __cuda_callable__ ( Index i, Index j )
         {
            typename MeshFunctionType::MeshType::Cell entity(mesh);
            entity.getCoordinates().x() = beginx + j;
            entity.getCoordinates().y() = beginy + i;				
            entity.refresh();
            if( ! isBoundary || ! mask || ( *mask )[ entity.getIndex() ] )
            {
               if( tobuffer )
                  buffer[ i * sizex + j ] = meshFunctionData[ entity.getIndex() ];
               else
                  meshFunctionData[ entity.getIndex() ] = buffer[ i * sizex + j ];
            }
         };
         ParallelFor2D< Device >::exec( 0, 0, sizey, sizex, kernel );     
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
         const Containers::StaticVector<3,Index>& begin,
         const Containers::StaticVector<3,Index>& size,
         bool tobuffer)
      {

         Index beginx=begin.x();
         Index beginy=begin.y();
         Index beginz=begin.z();
         Index sizex=size.x();
         Index sizey=size.y();
         Index sizez=size.z();

         auto mesh=meshFunction.getMesh();
         RealType * meshFunctionData=meshFunction.getData().getData();
         const typename MaskPointer::ObjectType* mask( nullptr );
         if( maskPointer )
            mask = &maskPointer.template getData< Device >();         
         auto kernel = [ tobuffer, mesh, mask, buffer, isBoundary, meshFunctionData, beginx, sizex, beginy, sizey, beginz] __cuda_callable__ ( Index k, Index i, Index j )
         {
            typename MeshFunctionType::MeshType::Cell entity(mesh);
            entity.getCoordinates().x() = beginx + j;
            entity.getCoordinates().z() = beginz + k;
            entity.getCoordinates().y() = beginy + i;
            entity.refresh();
            if( ! isBoundary || ! mask || ( *mask )[ entity.getIndex() ] )
            {
               if( tobuffer )
                  buffer[ k * sizex * sizey + i * sizex + j ] = 
                     meshFunctionData[ entity.getIndex() ];
               else
                  meshFunctionData[ entity.getIndex() ] = buffer[ k * sizex * sizey + i * sizex + j ];
            }
         };
         ParallelFor3D< Device >::exec( 0, 0, 0, sizez, sizey, sizex, kernel ); 
      };
};


} // namespace DistributedMeshes
} // namespace Meshes
} // namespace TNL
