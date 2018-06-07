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

#ifdef HAVE_MPI
    #ifdef MPIIO
        #include <mpi.h>
    #endif
#endif

#include <TNL/File.h>
#include <TNL/Communicators/MpiCommunicator.h>
#include <TNL/Meshes/DistributedMeshes/DistributedMesh.h>
#include <TNL/Meshes/DistributedMeshes/CopyEntitiesHelper.h>
#include <TNL/Functions/MeshFunction.h>

namespace TNL {
namespace Meshes {   
namespace DistributedMeshes {

enum DistrGridIOTypes { Dummy = 0 , LocalCopy = 1, MpiIO=2 };
    
template<typename MeshFunctionType,
         DistrGridIOTypes type = LocalCopy> 
class DistributedGridIO
{
};

template<typename MeshFunctionType> 
class DistributedGridIO<MeshFunctionType,Dummy>
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


/*
 * This variant cerate copy of MeshFunction but smaller, reduced to local entities, without overlap. 
 * It is slow and has high RAM consumption
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
    
    static bool save(const String& fileName, MeshFunctionType &meshFunction)
    {
        auto *distrGrid=meshFunction.getMesh().getDistributedMesh();
        
        if(distrGrid==NULL) //not distributed
        {
            return meshFunction.save(fileName);
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
        
        File meshFile;
        meshFile.open( fileName+String("-mesh-")+distrGrid->printProcessCoords()+String(".tnl"),IOMode::write);
        newMesh->save( meshFile );
        meshFile.close();

        VectorType newDof(newMesh-> template getEntitiesCount< typename MeshType::Cell >());

        MeshFunctionType newMeshFunction;
        newMeshFunction.bind(newMesh,newDof);        

        CoordinatesType zeroCoord;
        zeroCoord.setValue(0);

        CopyEntitiesHelper<MeshFunctionType>::Copy(meshFunction,newMeshFunction,localBegin,zeroCoord,localSize);

        File file;
        file.open( fileName+String("-")+distrGrid->printProcessCoords()+String(".tnl"), IOMode::write );
        bool ret=newMeshFunction.save(file);
        file.close();

        return ret;
        
    };
            
    static bool load(const String& fileName,MeshFunctionType &meshFunction) 
    {
        auto *distrGrid=meshFunction.getMesh().getDistributedMesh();
        if(distrGrid==NULL) //not distributed
        {
            return meshFunction.boundLoad(fileName);
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

        File file;
        file.open( fileName+String("-")+distrGrid->printProcessCoords()+String(".tnl"), IOMode::read );
        bool result=newMeshFunction.boundLoad(file);
        file.close();
        CopyEntitiesHelper<MeshFunctionType>::Copy(newMeshFunction,meshFunction,zeroCoord,localBegin,localSize);
        
        return result;
    };
    
};

/*
 * Save distributed data into single file without overlaps using MPIIO and MPI datatypes, 
 * EXPLOSIVE: works with only Grids and MPI
 * BAD IMPLEMENTTION creating MPI-Types at every save! -- I dont want contamine more places by MPI..
 */

#ifdef HAVE_MPI
#ifdef MPIIO  
template<typename MeshFunctionType> 
class DistributedGridIO<MeshFunctionType,MpiIO>
{
   public:

      using RealType = typename MeshFunctionType::RealType;
      typedef typename MeshFunctionType::MeshType MeshType;
      typedef typename MeshFunctionType::MeshType::CoordinatesType CoordinatesType;
      typedef typename MeshFunctionType::MeshType::PointType PointType;
      typedef typename MeshFunctionType::VectorType VectorType;
      //typedef DistributedGrid< MeshType,MeshFunctionType::getMeshDimension()> DistributedGridType;
    
    static bool save(const String& fileName, MeshFunctionType &meshFunction)
    {
     
        auto *distrGrid=meshFunction.getMesh().getDistributedMesh();
        
        if(distrGrid==NULL) //not distributed
        {
            return meshFunction.save(fileName);
        }

       MPI_Datatype ftype;
       MPI_Datatype atype;
       int dataCount=CreateDataTypes(distrGrid,&ftype,&atype);

       RealType* data=meshFunction.getData().getData();

       //write 
       MPI_File file;
       MPI_File_open( MPI_COMM_WORLD,
                      const_cast< char* >( fileName.getString() ),
                      MPI_MODE_CREATE | MPI_MODE_WRONLY,
                      MPI_INFO_NULL,
                      &file);

       int headerSize=0;

       if(Communicators::MpiCommunicator::GetRank()==0)
       {
            headerSize=writeMeshFunctionHeader(file,meshFunction,dataCount);
       }
       MPI_Bcast(&headerSize, 1, MPI_INT,0, MPI_COMM_WORLD);

       if( std::is_same< RealType, double >::value)
         MPI_File_set_view(file,headerSize,MPI_DOUBLE,ftype,"native",MPI_INFO_NULL);
       if( std::is_same< RealType, float >::value)
         MPI_File_set_view(file,headerSize,MPI_FLOAT,ftype,"native",MPI_INFO_NULL);
       
       MPI_Status wstatus;

       MPI_File_write(file,data,1,atype,&wstatus);

       MPI_File_close(&file);

       MPI_Type_free(&atype);
       MPI_Type_free(&ftype);
       return true;
    };

    template<typename DitsributedGridType>
    static int CreateDataTypes(DitsributedGridType *distrGrid,MPI_Datatype *ftype,MPI_Datatype *atype)
    {
        int dim=distrGrid->getMeshDimension();

        int fstarts[dim];
        int flsize[dim];
        int fgsize[dim];
        
        hackArray(dim,fstarts,distrGrid->getGlobalBegin().getData());
        hackArray(dim,flsize,distrGrid->getLocalSize().getData());
        hackArray(dim,fgsize,distrGrid->getGlobalSize().getData());

        MPI_Type_create_subarray(dim,
            fgsize,flsize,fstarts,
            MPI_ORDER_C,MPI_DOUBLE,ftype); //TYP
        MPI_Type_commit(ftype);

       int agsize[dim];
       int alsize[dim];
       int astarts[dim]; 

       hackArray(dim,astarts,distrGrid->getLocalBegin().getData());
       hackArray(dim,alsize,distrGrid->getLocalSize().getData());
       hackArray(dim,agsize,distrGrid->getLocalGridSize().getData());

       MPI_Type_create_subarray(dim,
            agsize,alsize,astarts,
            MPI_ORDER_C,MPI_DOUBLE,atype); //TYP
       MPI_Type_commit(atype);

        int dataCount=1;
        for(int i=0;i<dim;i++)
            dataCount*=fgsize[i];

        return dataCount;

    }

    template<typename Index>
    static void hackArray(int dim, int* out, Index* in)
    {
        if(dim==1)
        {
            out[0]=in[0];
        }

        if(dim==2)
        {
           out[1]=in[0];
           out[0]=in[1];
        }

        if(dim==3)
        {
           out[0]=in[2];
           out[1]=in[1];
           out[2]=in[0];
        }
    }

    static unsigned int writeMeshFunctionHeader(MPI_File file,MeshFunctionType &meshFunction, int length )
    {

        unsigned int size=0;
        int count;
        MPI_Status wstatus;

        //Magic
        MPI_File_write( file, const_cast< void* >( ( const void* ) "TNLMN" ), 5, MPI_CHAR,&wstatus );
        MPI_Get_count(&wstatus,MPI_CHAR,&count);
        size+=count*sizeof(char);

        //Meshfunction type
        String meshFunctionSerializationType=meshFunction.getSerializationTypeVirtual();
        int meshFunctionSerializationTypeLength=meshFunctionSerializationType.getLength();
        MPI_File_write(file,&meshFunctionSerializationTypeLength,1,MPI_INT,&wstatus);
        MPI_Get_count(&wstatus,MPI_INT,&count);
        size+=count*sizeof(int);
        MPI_File_write(file,meshFunctionSerializationType.getString(),meshFunctionSerializationType.getLength(),MPI_CHAR,&wstatus);
        MPI_Get_count(&wstatus,MPI_CHAR,&count);
        size+=count*sizeof(char);

        //Magic
        MPI_File_write( file, const_cast< void* >( ( const void* ) "TNLMN" ),5,MPI_CHAR,&wstatus);
        MPI_Get_count(&wstatus,MPI_CHAR,&count);
        size+=count*sizeof(char);
        //Vector Type
        String dataSerializationType=meshFunction.getData().getSerializationTypeVirtual();
        int dataSerializationTypeLength=dataSerializationType.getLength();
        MPI_File_write(file,&dataSerializationTypeLength,1,MPI_INT,&wstatus);
        MPI_Get_count(&wstatus,MPI_INT,&count);
        size+=count*sizeof(int);
        MPI_File_write(file,dataSerializationType.getString(),dataSerializationType.getLength(),MPI_CHAR,&wstatus);
        MPI_Get_count(&wstatus,MPI_CHAR,&count);
        size+=count*sizeof(char);
        //Data count
        MPI_File_write(file,&(length),1,MPI_INT,&wstatus);
        MPI_Get_count(&wstatus,MPI_INT,&count);
        size+=count*sizeof(int);

        return size;
    };
            
    /* Funky bomb - no checks - only dirty load */
    static bool load(const String& fileName,MeshFunctionType &meshFunction) 
    {
        auto *distrGrid=meshFunction.getMesh().getDistributedMesh();
        if(distrGrid==NULL) //not distributed
        {
            return meshFunction.boundLoad(fileName);
        }

       MPI_Datatype ftype;
       MPI_Datatype atype;
       int dataCount=CreateDataTypes(distrGrid,&ftype,&atype);

       double * data=meshFunction.getData().getData();//TYP

       //write 
       MPI_File file;
       MPI_File_open( MPI_COMM_WORLD,
                      const_cast< char* >( fileName.getString() ),
                      MPI_MODE_RDONLY,
                      MPI_INFO_NULL,
                      &file );
       
       int headerSize=0;

       if(Communicators::MpiCommunicator::GetRank()==0)
       {
            headerSize=readMeshFunctionHeader(file,meshFunction,dataCount);
       }
       MPI_Bcast(&headerSize, 1, MPI_INT,0, MPI_COMM_WORLD);
       
       if(headerSize<0)
            return false;

       MPI_File_set_view(file,headerSize,MPI_DOUBLE,ftype,"native",MPI_INFO_NULL);//TYP
       MPI_Status wstatus;
       MPI_File_read(file,(void*)data,1,atype,&wstatus);
       MPI_File_close(&file);
        
       MPI_Type_free(&atype);
       MPI_Type_free(&ftype);

       return true;
    };

    //tak mohlo by to něco kontrolovat...ale nic to nekontroluje
    static int readMeshFunctionHeader(MPI_File file,MeshFunctionType &meshFunction, int length)
    {
        MPI_Status rstatus;
        char buffer[255];
        int size=0;
        int count=0;

        MPI_File_read(file, (void *)buffer,5, MPI_CHAR, &rstatus);//MAGIC
        size+=5*sizeof(char);
        MPI_File_read(file, (void *)&count,1, MPI_INT, &rstatus);//SIZE OF DATATYPE
        size+=1*sizeof(int);
        MPI_File_read(file, (void *)&buffer,count, MPI_CHAR, &rstatus);//DATATYPE
        size+=count*sizeof(char);

        MPI_File_read(file, (void *)buffer,5, MPI_CHAR, &rstatus);//MAGIC
        size+=5*sizeof(char);
        MPI_File_read(file, (void *)&count,1, MPI_INT, &rstatus);//SIZE OF DATATYPE
        size+=1*sizeof(int);
        MPI_File_read(file, (void *)&buffer,count, MPI_CHAR, &rstatus);//DATATYPE
        size+=count*sizeof(char);
        MPI_File_read(file, (void *)&count,1, MPI_INT, &rstatus);//DATACOUNT
        size+=1*sizeof(int);
        
        if(count!=length)
        {
            std::cerr<<"Chyba načítání MeshFunction, délka dat v souboru neodpovídá očekávané délce" << std::endl;
            size=-1;
        }

        return size;
    };
    
};
#endif
#endif
}
}
}
