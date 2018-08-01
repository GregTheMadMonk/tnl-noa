/***************************************************************************
                          DistributedGrid.h  -  description
                             -------------------
    begin                : October 5, 2017
    copyright            : (C) 2017 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <iostream>

#include <TNL/File.h>
#include <TNL/Communicators/MpiCommunicator.h>
#include <TNL/Meshes/DistributedMeshes/DistributedMesh.h>
#include <TNL/Meshes/DistributedMeshes/CopyEntitiesHelper.h>
#include <TNL/Functions/MeshFunction.h>
#include <TNL/Devices/Host.h>
#include <TNL/Devices/Cuda.h>

namespace TNL {
namespace Meshes {   
namespace DistributedMeshes {

enum DistrGridIOTypes { Dummy = 0 , LocalCopy = 1, MpiIO=2 };
    
template<typename MeshFunctionType,
         DistrGridIOTypes type = LocalCopy,
         typename Device=typename MeshFunctionType::DeviceType> 
class DistributedGridIO
{
};

template<typename MeshFunctionType,
         typename Device> 
class DistributedGridIO<MeshFunctionType,Dummy,Device>
{
    bool save(const String& fileName, MeshFunctionType &meshFunction)
    {
        return true;
    };
            
    bool load(const String& fileName, MeshFunctionType &meshFunction) 
    {
        return true;
    };
};





}
}
}

#include <TNL/Meshes/DistributedMeshes/DistributedGridIO_MeshFunction.h>
