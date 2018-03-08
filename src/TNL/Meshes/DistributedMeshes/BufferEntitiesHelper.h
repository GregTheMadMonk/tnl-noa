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
           int dim,
           typename RealType,
           typename Device >
class BufferEntitiesHelper
{
};

//======================================== 1D ====================================================

//host
template < typename MeshFunctionType, typename RealType, typename Device >
class BufferEntitiesHelper<MeshFunctionType,1,RealType,Device>
{
    public:
    static void BufferEntities(MeshFunctionType meshFunction, RealType * buffer, int beginx, int sizex,bool tobuffer)
    {
        auto mesh = meshFunction.getMesh();
        RealType* meshFunctionData = meshFunction.getData().getData();
        auto kernel = [tobuffer, mesh, buffer, meshFunctionData, beginx] __cuda_callable__ ( int j )
        {
            typename MeshFunctionType::MeshType::Cell entity(mesh);
            entity.getCoordinates().x()=beginx+j;
            entity.refresh();
            if(tobuffer)
                buffer[j]=meshFunctionData[entity.getIndex()];
            else
                meshFunctionData[entity.getIndex()]=buffer[j];
        };
        ParallelFor< Device >::exec( 0, sizex, kernel );
    };  
};


//======================================== 2D ====================================================
template <typename MeshFunctionType, typename RealType, typename Device > 
class BufferEntitiesHelper<MeshFunctionType,2,RealType,Device>
{
    public:
    static void BufferEntities(MeshFunctionType meshFunction, RealType * buffer, int beginx, int beginy, int sizex, int sizey,bool tobuffer)
    {
        auto mesh=meshFunction.getMesh();
        RealType *meshFunctionData=meshFunction.getData().getData();
        auto kernel = [tobuffer, mesh, buffer, meshFunctionData, beginx, sizex, beginy] __cuda_callable__ ( int i, int j )
        {
            typename MeshFunctionType::MeshType::Cell entity(mesh);
            entity.getCoordinates().x()=beginx+j;
            entity.getCoordinates().y()=beginy+i;				
            entity.refresh();
            if(tobuffer)
                    buffer[i*sizex+j]=meshFunctionData[entity.getIndex()];
            else
                    meshFunctionData[entity.getIndex()]=buffer[i*sizex+j];
        };
        
        ParallelFor2D< Device >::exec( 0, 0, sizey, sizex, kernel );       
        
    };
};


//======================================== 3D ====================================================
template <typename MeshFunctionType, typename RealType, typename Device>
class BufferEntitiesHelper<MeshFunctionType,3,RealType,Device>
{
    public:
    static void BufferEntities(MeshFunctionType meshFunction, RealType * buffer, int beginx, int beginy, int beginz, int sizex, int sizey, int sizez, bool tobuffer)
    {

        auto mesh=meshFunction.getMesh();
        RealType * meshFunctionData=meshFunction.getData().getData();
        auto kernel = [tobuffer, mesh, buffer, meshFunctionData, beginx, sizex, beginy, sizey, beginz] __cuda_callable__ ( int k, int i, int j )
        {
            typename MeshFunctionType::MeshType::Cell entity(mesh);
            entity.getCoordinates().x()=beginx+j;
            entity.getCoordinates().z()=beginz+k;
            entity.getCoordinates().y()=beginy+i;
            entity.refresh();
            if(tobuffer)
                    buffer[k*sizex*sizey+i*sizex+j]=meshFunctionData[entity.getIndex()];
            else
                    meshFunctionData[entity.getIndex()]=buffer[k*sizex*sizey+i*sizex+j];
        };

        ParallelFor3D< Device >::exec( 0, 0, 0, sizez, sizey, sizex, kernel ); 

        /*for(int k=0;k<sizez;k++)
        {
            for(int i=0;i<sizey;i++)
            {
                for(int j=0;j<sizex;j++)
                {
                        kernel(k,i,j);
                }
            }
        }*/
    };
};


} // namespace DistributedMeshes
} // namespace Meshes
} // namespace TNL
