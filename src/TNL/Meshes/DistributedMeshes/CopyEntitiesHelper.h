/***************************************************************************
                          CopyEntitiesHelper.h  -  description
                             -------------------
    begin                : March 8, 2018
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

template<typename MeshFunctionType,
         int dim=MeshFunctionType::getMeshDimension()>
class CopyEntitiesHelper
{
    public:
    typedef typename MeshFunctionType::MeshType::CoordinatesType CoordinatesType;
    static void Copy(MeshFunctionType &from, MeshFunctionType &to, CoordinatesType &fromBegin, CoordinatesType &toBegin, CoordinatesType &size)
    {
    }

};


template<typename MeshFunctionType>
class CopyEntitiesHelper<MeshFunctionType, 1>
{
    public:
    typedef typename MeshFunctionType::MeshType::CoordinatesType CoordinatesType;
    typedef typename MeshFunctionType::MeshType::Cell Cell;

    static void Copy(MeshFunctionType &from, MeshFunctionType &to, CoordinatesType &fromBegin, CoordinatesType &toBegin, CoordinatesType &size)
    {        
        auto toData=to.getData().getData();
        auto fromData=from.getData().getData();
        auto fromMesh=from.getMesh();
        auto toMesh=to.getMesh();
        auto kernel = [fromData,toData, fromMesh, toMesh, fromBegin, toBegin] __cuda_callable__ ( int i )
        {
            Cell fromEntity(fromMesh);
            Cell toEntity(toMesh);
            toEntity.getCoordinates().x()=toBegin.x()+i;            
            toEntity.refresh();
            fromEntity.getCoordinates().x()=fromBegin.x()+i;            
            fromEntity.refresh();
            toData[toEntity.getIndex()]=fromData[fromEntity.getIndex()];
        };
        ParallelFor< typename MeshFunctionType::MeshType::DeviceType >::exec( 0, size.x(), kernel );

    }

};


template<typename MeshFunctionType>

class CopyEntitiesHelper<MeshFunctionType,2>
{
    public:
    typedef typename MeshFunctionType::MeshType::CoordinatesType CoordinatesType;
    typedef typename MeshFunctionType::MeshType::Cell Cell;

    static void Copy(MeshFunctionType &from, MeshFunctionType &to, CoordinatesType &fromBegin, CoordinatesType &toBegin, CoordinatesType &size)
    {
        auto toData=to.getData().getData();
        auto fromData=from.getData().getData();
        auto fromMesh=from.getMesh();
        auto toMesh=to.getMesh();
        auto kernel = [fromData,toData, fromMesh, toMesh, fromBegin, toBegin] __cuda_callable__ ( int j, int i )
        {
            Cell fromEntity(fromMesh);
            Cell toEntity(toMesh);
            toEntity.getCoordinates().x()=toBegin.x()+i;
            toEntity.getCoordinates().y()=toBegin.y()+j;            
            toEntity.refresh();
            fromEntity.getCoordinates().x()=fromBegin.x()+i;
            fromEntity.getCoordinates().y()=fromBegin.y()+j;            
            fromEntity.refresh();
            toData[toEntity.getIndex()]=fromData[fromEntity.getIndex()];
        };
        ParallelFor2D< typename MeshFunctionType::MeshType::DeviceType >::exec( 0,0,size.y(), size.x(), kernel );

    }

};


template<typename MeshFunctionType>
class CopyEntitiesHelper<MeshFunctionType,3>
{
    public:
    typedef typename MeshFunctionType::MeshType::CoordinatesType CoordinatesType;
    typedef typename MeshFunctionType::MeshType::Cell Cell;

    static void Copy(MeshFunctionType &from, MeshFunctionType &to, CoordinatesType &fromBegin, CoordinatesType &toBegin, CoordinatesType &size)
    {
        auto toData=to.getData().getData();
        auto fromData=from.getData().getData();
        auto fromMesh=from.getMesh();
        auto toMesh=to.getMesh();
        auto kernel = [fromData,toData, fromMesh, toMesh, fromBegin, toBegin] __cuda_callable__ ( int k,int j, int i )
        {
            Cell fromEntity(fromMesh);
            Cell toEntity(toMesh);
            toEntity.getCoordinates().x()=toBegin.x()+i;
            toEntity.getCoordinates().y()=toBegin.y()+j;
            toEntity.getCoordinates().z()=toBegin.z()+k;                                
            toEntity.refresh();
            fromEntity.getCoordinates().x()=fromBegin.x()+i;
            fromEntity.getCoordinates().y()=fromBegin.y()+j;
            fromEntity.getCoordinates().z()=fromBegin.z()+k;            
            fromEntity.refresh();
            toData[toEntity.getIndex()]=fromData[fromEntity.getIndex()];
        };
        ParallelFor3D< typename MeshFunctionType::MeshType::DeviceType >::exec( 0,0,0,size.z() ,size.y(), size.x(), kernel );
    }
};




} // namespace DistributedMeshes
} // namespace Meshes
} // namespace TNL
