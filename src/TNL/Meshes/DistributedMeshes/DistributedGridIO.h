/***************************************************************************
                          DistributedGrid.h  -  description
                             -------------------
    begin                : October 5, 2017
    copyright            : (C) 2017 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/File.h>
#include <TNL/Meshes/DistributedMeshes/DistributedMesh.h>
#include <TNL/Meshes/DistributedMeshes/CopyEntitiesHelper.h>
#include <TNL/Functions/MeshFunction.h>


#include <iostream>

namespace TNL {
namespace Meshes {   
namespace DistributedMeshes {

enum DistrGridIOTypes { Dummy = 0 , LocalCopy = 1 };
    
template<typename MeshFunctionType,
         DistrGridIOTypes type = LocalCopy> 
class DistributedGridIO
{
};

template<typename MeshFunctionType> 
class DistributedGridIO<MeshFunctionType,Dummy>
{
    bool save(File &file,MeshFunctionType meshFunction)
    {
        return true;
    };
            
    bool load(File &file,MeshFunctionType &meshFunction) 
    {
        return true;
    };
};


/*
 * This variant cerate copy of MeshFunction but smaler, reduced to local entites, without overlap. 
 * It slow and has high RAM consupation
 */
template<typename MeshFunctionType> 
class DistributedGridIO<MeshFunctionType,LocalCopy>
{

    public:

    typedef typename MeshFunctionType::MeshType MeshType;
    typedef typename MeshFunctionType::MeshType::CoordinatesType CoordinatesType;
    typedef typename MeshFunctionType::MeshType::PointType PointType;
    typedef typename MeshFunctionType::VectorType VectorType;
    //typedef DistributedGrid< MeshType,MeshFunctionType::getMeshDimension()> DistributedGridType;
    
    static bool save(File &file,MeshFunctionType &meshFunction)
    {
        auto *distrGrid=meshFunction.getMesh().GetDistMesh();
        
        if(distrGrid==NULL) //not distributed
        {
            return meshFunction.save(file);
        }

        MeshType mesh=meshFunction.getMesh();
        
        PointType spaceSteps=mesh.getSpaceSteps();
        PointType origin=mesh.getOrigin();
                
        CoordinatesType localSize=distrGrid->getLocalSize();
        CoordinatesType localBegin=distrGrid->getLocalBegin();
 
        SharedPointer<MeshType> newMesh;
        newMesh->setDimensions(localSize);
        newMesh->setSpaceSteps(spaceSteps);
        newMesh->setOrigin(origin+TNL::Containers::tnlDotProduct(spaceSteps,localBegin));
        
        VectorType newDof(newMesh-> template getEntitiesCount< typename MeshType::Cell >());

        MeshFunctionType newMeshFunction;
        newMeshFunction.bind(newMesh,newDof);        

        CoordinatesType zeroCoord;
        zeroCoord.setValue(0);

        CopyEntitiesHelper<MeshFunctionType>::Copy(meshFunction,newMeshFunction,localBegin,zeroCoord,localSize);
        return newMeshFunction.save(file);
        
    };
            
    static bool load(File &file,MeshFunctionType &meshFunction) 
    {
        auto *distrGrid=meshFunction.getMesh().GetDistMesh();
        if(distrGrid==NULL) //not distributed
        {
            return meshFunction.boundLoad(file);
        }

        MeshType mesh=meshFunction.getMesh();
        
        PointType spaceSteps=mesh.getSpaceSteps();
        PointType origin=mesh.getOrigin();
                
        CoordinatesType localSize=distrGrid->getLocalSize();
        CoordinatesType localBegin=distrGrid->getLocalBegin();

        SharedPointer<MeshType> newMesh;
        newMesh->setDimensions(localSize);
        newMesh->setSpaceSteps(spaceSteps);
        newMesh->setOrigin(origin+TNL::Containers::tnlDotProduct(spaceSteps,localBegin));
        
        VectorType newDof(newMesh-> template getEntitiesCount< typename MeshType::Cell >());
        MeshFunctionType newMeshFunction;
        newMeshFunction.bind(newMesh,newDof); 

        CoordinatesType zeroCoord;
        zeroCoord.setValue(0);        

        bool result=newMeshFunction.boundLoad(file);
        CopyEntitiesHelper<MeshFunctionType>::Copy(newMeshFunction,meshFunction,zeroCoord,localBegin,localSize);
        
        return result;
    };
    
};

}
}
}
