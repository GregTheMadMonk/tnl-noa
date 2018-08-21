/***************************************************************************
                          CutMeshFunction.h  -  description
                             -------------------
    begin                : July 12, 2018
    copyright            : (C) 2018 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/Functions/MeshFunction.h>
#include <TNL/StaticVectorFor.h>
#include <TNL/Containers/StaticVector.h>

namespace TNL {
namespace Functions {  
template <  typename CommunicatorType,
            typename MeshFunctionType,
            typename OutMesh,
            typename OutDof,
            int outDimension=OutMesh::getMeshDimension(),
            int codimension=MeshFunctionType::getMeshDimension()-OutMesh::getMeshDimension()>
class CutMeshFunction
{
  public:
    static bool Cut(MeshFunctionType &inputMeshFunction,
                    OutMesh &outMesh, 
                    OutDof &outData,
                    Containers::StaticVector<outDimension, int> savedDimensions, 
                    Containers::StaticVector<codimension,int> reducedDimensions, 
                    Containers::StaticVector<codimension,typename MeshFunctionType::IndexType> fixedIndexs )
    {
        bool inCut;
        Containers::StaticVector<codimension,typename MeshFunctionType::IndexType> localFixedIndexs;

        auto fromData=inputMeshFunction.getData().getData();
        auto fromMesh=inputMeshFunction.getMesh();

        //Set-up Grid
        auto fromDistributedGrid=fromMesh.getDistributedMesh();
        if(fromDistributedGrid!=nullptr)
        {
            auto toDistributedGrid=outMesh.getDistributedMesh();
            TNL_ASSERT_TRUE(toDistributedGrid!=nullptr,"You are trying cut distributed meshfunction, but output grid is not set up for distribution");

            inCut=toDistributedGrid-> template SetupByCut<CommunicatorType>(*fromDistributedGrid,savedDimensions,reducedDimensions,fixedIndexs);
            if(inCut)
            {
               toDistributedGrid->setupGrid(outMesh);
               for(int i=0;i<codimension;i++)
                   localFixedIndexs[i]=fixedIndexs[i]-fromDistributedGrid->getGlobalBegin()[reducedDimensions[i]];
            }
        }
        else
        {
            typename OutMesh::PointType outOrigin;
            typename OutMesh::PointType outProportions;
            typename OutMesh::CoordinatesType outDimensions; 

            for(int i=0; i<outDimension;i++)
            {
                outOrigin[i]=fromMesh.getOrigin()[savedDimensions[i]];
                outProportions[i]=fromMesh.getProportions()[savedDimensions[i]];
                outDimensions[i]=fromMesh.getDimensions()[savedDimensions[i]];
            }
            
            outMesh.setDimensions(outDimensions);
            outMesh.setDomain(outOrigin,outProportions);
            
            inCut=true;
            localFixedIndexs=fixedIndexs;
            
        }

        //copy data
        if(inCut)
        {
            outData.setSize(outMesh.template getEntitiesCount< typename OutMesh::Cell >());
            auto kernel = [&fromData, &fromMesh, &outData, &outMesh, &savedDimensions,&localFixedIndexs,&reducedDimensions ] ( typename OutMesh::CoordinatesType index )
            {

                typename MeshFunctionType::MeshType::Cell fromEntity(fromMesh);
                typename OutMesh::Cell outEntity(outMesh);

                for(int j=0;j<outDimension;j++)
                {
                    fromEntity.getCoordinates()[savedDimensions[j]]=index[j];
                    outEntity.getCoordinates()[j]=index[j];
                }

                for(int j=0; j<codimension;j++)
                    fromEntity.getCoordinates()[reducedDimensions[j]]=localFixedIndexs[j];

                fromEntity.refresh();
                outEntity.refresh();
                outData[outEntity.getIndex()]=fromData[fromEntity.getIndex()];
            };


            typename OutMesh::CoordinatesType starts;
            starts.setValue(0);
            StaticVectorFor::exec(starts,outMesh.getDimensions(),kernel);
        }

        return inCut;
    } 
};

} // namespace Functions
} // namespace TNL

