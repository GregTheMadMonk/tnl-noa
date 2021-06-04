/***************************************************************************
                          DistributedGridIO_NeshFunction.h  -  description
                             -------------------
    begin                : October 5, 2017
    copyright            : (C) 2017 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/Functions/MeshFunction.h>
#include <TNL/Functions/MeshFunctionView.h>
#include <TNL/MPI/getDataType.h>

namespace TNL {
namespace Meshes {
namespace DistributedMeshes {


/*
 * This variant cerate copy of MeshFunction but smaller, reduced to local entities, without overlap.
 * It is slow and has high RAM consumption
 */
template< typename MeshFunction,
          int Dimension,
          typename MeshReal,
          typename Device,
          typename Index >
class DistributedGridIO<
   MeshFunction,
   LocalCopy,
   Meshes::Grid< Dimension, MeshReal, Device, Index > >
{
   public:
      using MeshType = Meshes::Grid< Dimension, MeshReal, Device, Index >;
      using MeshFunctionType = MeshFunction;
      using MeshFunctionViewType = Functions::MeshFunctionView< MeshType, MeshFunction::getEntitiesDimension(), typename MeshFunction::RealType >;
      using CoordinatesType = typename MeshFunctionType::MeshType::CoordinatesType;
      using PointType = typename MeshFunctionType::MeshType::PointType;
      using VectorType = Containers::Vector< typename MeshFunctionType::RealType, typename MeshFunctionType::DeviceType, typename MeshFunctionType::IndexType >;
      //typedef DistributedGrid< MeshType,MeshFunctionType::getMeshDimension()> DistributedGridType;

      static bool save(const String& fileName, MeshFunctionType &meshFunction)
      {
         auto *distrGrid=meshFunction.getMesh().getDistributedMesh();

         if(distrGrid==NULL) //not distributed
         {
            meshFunction.save(fileName);
            return true;
         }

         const MeshType& mesh=meshFunction.getMesh();

         PointType spaceSteps=mesh.getSpaceSteps();
         PointType origin=mesh.getOrigin();

         CoordinatesType localSize=distrGrid->getLocalSize();
         CoordinatesType localBegin=distrGrid->getLocalBegin();

         Pointers::SharedPointer<MeshType> newMesh;
         newMesh->setDimensions(localSize);
         newMesh->setSpaceSteps(spaceSteps);
         CoordinatesType newOrigin;
         newMesh->setOrigin(origin+spaceSteps*localBegin);

         // FIXME: save was removed from Grid (but this is probably just for debugging...)
//         File meshFile;
//         meshFile.open( fileName+String("-mesh-")+distrGrid->printProcessCoords()+String(".tnl"), std::ios_base::out );
//         newMesh->save( meshFile );
//         meshFile.close();

         VectorType newDof(newMesh-> template getEntitiesCount< typename MeshType::Cell >());

         MeshFunctionViewType newMeshFunction;
         newMeshFunction.bind(newMesh,newDof);

         CoordinatesType zeroCoord;
         zeroCoord.setValue(0);

         CopyEntitiesHelper<MeshFunctionViewType>::Copy(meshFunction,newMeshFunction,localBegin,zeroCoord,localSize);

         File file;
         file.open( fileName+String("-")+distrGrid->printProcessCoords()+String(".tnl"), std::ios_base::out );
         newMeshFunction.save(file);
         file.close();

         return true;

      };

    static bool load(const String& fileName,MeshFunctionType &meshFunction)
    {
        auto *distrGrid=meshFunction.getMesh().getDistributedMesh();
        if(distrGrid==NULL) //not distributed
        {
            meshFunction.boundLoad(fileName);
            return true;
        }

        const MeshType& mesh=meshFunction.getMesh();

        PointType spaceSteps=mesh.getSpaceSteps();
        PointType origin=mesh.getOrigin();

        CoordinatesType localSize=distrGrid->getLocalSize();
        CoordinatesType localBegin=distrGrid->getLocalBegin();

        Pointers::SharedPointer<MeshType> newMesh;
        newMesh->setDimensions(localSize);
        newMesh->setSpaceSteps(spaceSteps);
        CoordinatesType newOrigin;
        newMesh->setOrigin(origin+spaceSteps*localBegin);

        VectorType newDof(newMesh-> template getEntitiesCount< typename MeshType::Cell >());
        MeshFunctionType newMeshFunction;
        newMeshFunction.bind(newMesh,newDof);

        CoordinatesType zeroCoord;
        zeroCoord.setValue(0);

        File file;
        file.open( fileName+String("-")+distrGrid->printProcessCoords()+String(".tnl"), std::ios_base::in );
        newMeshFunction.boundLoad(file);
        file.close();
        CopyEntitiesHelper<MeshFunctionType>::Copy(newMeshFunction,meshFunction,zeroCoord,localBegin,localSize);

        return true;
    };

};

/*
 * Save distributed data into single file without overlaps using MPIIO and MPI datatypes,
 * EXPLOSIVE: works with only Grids and MPI
 * BAD IMPLEMENTTION creating MPI-Types at every save! -- I dont want contamine more places by MPI..
 */

#ifdef HAVE_MPI
template<typename MeshFunctionType>
class DistributedGridIO_MPIIOBase
{
   public:

      using RealType = typename MeshFunctionType::RealType;
      typedef typename MeshFunctionType::MeshType MeshType;
      typedef typename MeshFunctionType::MeshType::CoordinatesType CoordinatesType;
      typedef typename MeshFunctionType::MeshType::PointType PointType;
      typedef typename MeshFunctionType::VectorType VectorType;
      //typedef DistributedGrid< MeshType,MeshFunctionType::getMeshDimension()> DistributedGridType;

    static bool save(const String& fileName, MeshFunctionType &meshFunction, RealType *data)
    {
		auto *distrGrid=meshFunction.getMesh().getDistributedMesh();

        if(distrGrid==NULL) //not distributed
        {
            meshFunction.save(fileName);
        }

       MPI_Comm group=distrGrid->getCommunicationGroup();

	   MPI_File file;
      int ok=MPI_File_open( group,
                      const_cast< char* >( fileName.getString() ),
                      MPI_MODE_CREATE | MPI_MODE_WRONLY,
                      MPI_INFO_NULL,
                      &file);
      if( ok != 0 )
         throw std::runtime_error("Open file falied");

		int written=save(file,meshFunction, data,0);

        MPI_File_close(&file);

		return written>0;

	};

    static int save(MPI_File &file, MeshFunctionType &meshFunction, RealType *data, int offset)
    {

       auto *distrGrid=meshFunction.getMesh().getDistributedMesh();
       MPI_Comm group=distrGrid->getCommunicationGroup();
       MPI_Datatype ftype;
       MPI_Datatype atype;
       int dataCount=CreateDataTypes(distrGrid,&ftype,&atype);

       int headerSize;

       MPI_File_set_view(file,0,MPI_BYTE,MPI_BYTE,"native",MPI_INFO_NULL);

       if(MPI::GetRank(group)==0)
       {
            MPI_File_seek(file,offset,MPI_SEEK_SET);
            headerSize=writeMeshFunctionHeader(file,meshFunction,dataCount);
       }
       MPI_Bcast(&headerSize, 1, MPI_INT,0, group);

	   offset +=headerSize;

       MPI_File_set_view(file,offset,
               TNL::MPI::getDataType<RealType>(),
               ftype,"native",MPI_INFO_NULL);

       MPI_Status wstatus;

       MPI_File_write(file,data,1,atype,&wstatus);

       MPI_Type_free(&atype);
       MPI_Type_free(&ftype);

       return headerSize+dataCount*sizeof(RealType); //size of written data

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
            MPI_ORDER_C,
            TNL::MPI::getDataType<RealType>(),
            ftype);

        MPI_Type_commit(ftype);

       int agsize[dim];
       int alsize[dim];
       int astarts[dim];

       hackArray(dim,astarts,distrGrid->getLocalBegin().getData());
       hackArray(dim,alsize,distrGrid->getLocalSize().getData());
       hackArray(dim,agsize,distrGrid->getLocalGridSize().getData());

       MPI_Type_create_subarray(dim,
            agsize,alsize,astarts,
            MPI_ORDER_C,
            TNL::MPI::getDataType<RealType>(),
            atype);
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
        MPI_File_write(file,const_cast< void* >( ( const void* ) meshFunctionSerializationType.getString() ),meshFunctionSerializationType.getLength(),MPI_CHAR,&wstatus);
        MPI_Get_count(&wstatus,MPI_CHAR,&count);
        size+=count*sizeof(char);

        //Magic
        MPI_File_write( file, const_cast< void* >( ( const void* ) "TNLMN" ),5,MPI_CHAR,&wstatus);
        MPI_Get_count(&wstatus,MPI_CHAR,&count);
        size+=count*sizeof(char);
        //Vector Type
//        String dataSerializationType=meshFunction.getData().getSerializationTypeVirtual();
        String dataSerializationType = Containers::detail::ArrayIO< RealType, typename MeshFunctionType::DeviceType, typename MeshFunctionType::IndexType >::getSerializationType();
        int dataSerializationTypeLength=dataSerializationType.getLength();
        MPI_File_write(file,&dataSerializationTypeLength,1,MPI_INT,&wstatus);
        MPI_Get_count(&wstatus,MPI_INT,&count);
        size+=count*sizeof(int);
        MPI_File_write( file, const_cast< void* >( ( const void* ) dataSerializationType.getString() ), dataSerializationType.getLength(), MPI_CHAR, &wstatus );
        MPI_Get_count(&wstatus,MPI_CHAR,&count);
        size+=count*sizeof(char);
        //Data count
        MPI_File_write(file,&(length),1,MPI_INT,&wstatus);
        MPI_Get_count(&wstatus,MPI_INT,&count);
        size+=count*sizeof(int);

        return size;
    };

   static bool load(const String& fileName,MeshFunctionType &meshFunction, RealType* data )
   {
      auto *distrGrid=meshFunction.getMesh().getDistributedMesh();
      if(distrGrid==NULL) //not distributed
      {
         meshFunction.boundLoad(fileName);
         return true;
      }

      MPI_Comm group=distrGrid->getCommunicationGroup();

      MPI_File file;
      if( MPI_File_open( group,
            const_cast< char* >( fileName.getString() ),
            MPI_MODE_RDONLY,
            MPI_INFO_NULL,
            &file ) != 0 )
      {
         std::cerr << "Unable to open file " << fileName.getString() << std::endl;
         return false;
      }
      bool ret= load(file, meshFunction, data,0)>0;

      MPI_File_close(&file);
      return ret;
   }

    /* Funky bomb - no checks - only dirty load */
    static int load(MPI_File &file,MeshFunctionType &meshFunction, RealType* data, int offset )
    {
       auto *distrGrid=meshFunction.getMesh().getDistributedMesh();

       MPI_Comm group=distrGrid->getCommunicationGroup();
       MPI_Datatype ftype;
       MPI_Datatype atype;
       int dataCount=CreateDataTypes(distrGrid,&ftype,&atype);

       MPI_File_set_view(file,0,MPI_BYTE,MPI_BYTE,"native",MPI_INFO_NULL);

       int headerSize=0;

       if(MPI::GetRank(group)==0)
       {
            MPI_File_seek(file,offset,MPI_SEEK_SET);
            headerSize=readMeshFunctionHeader(file,meshFunction,dataCount);
       }
       MPI_Bcast(&headerSize, 1, MPI_INT,0, group);

       if(headerSize<0)
            return false;

       offset+=headerSize;

       MPI_File_set_view(file,offset,
            TNL::MPI::getDataType<RealType>(),
            ftype,"native",MPI_INFO_NULL);
       MPI_Status wstatus;
       MPI_File_read(file,(void*)data,1,atype,&wstatus);

       MPI_Type_free(&atype);
       MPI_Type_free(&ftype);

       return headerSize+dataCount*sizeof(RealType); //size of readed data;
    };

    //it shoudl check some loaded files...... but chcek nothing...
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

template< typename MeshFunction,
          int Dimension,
          typename MeshReal,
          typename Index >
class DistributedGridIO<
   MeshFunction,
   MpiIO,
   Meshes::Grid< Dimension, MeshReal, Devices::Cuda, Index >,
   Devices::Cuda >
{
   public:
      using MeshType = Meshes::Grid< Dimension, MeshReal, Devices::Cuda, Index >;
      using MeshFunctionType = MeshFunction;

      static bool save(const String& fileName, MeshFunctionType &meshFunction)
      {
#ifdef HAVE_MPI
         if(MPI::isInitialized())//i.e. - isUsed
         {
            using HostVectorType = Containers::Vector<typename MeshFunctionType::RealType, Devices::Host, typename MeshFunctionType::IndexType >;
            HostVectorType hostVector;
            hostVector=meshFunction.getData();
            typename MeshFunctionType::RealType * data=hostVector.getData();
            return DistributedGridIO_MPIIOBase<MeshFunctionType>::save(fileName,meshFunction,data);
         }
#endif
         std::cout << "MPIIO can be used only when MPI is initialized." << std::endl;
         return false;
      };

      static bool load(const String& fileName,MeshFunctionType &meshFunction)
      {
#ifdef HAVE_MPI
         if(MPI::isInitialized())//i.e. - isUsed
         {
            using HostVectorType = Containers::Vector<typename MeshFunctionType::RealType, Devices::Host, typename MeshFunctionType::IndexType >;
            HostVectorType hostVector;
            hostVector.setLike(meshFunction.getData());
            auto* data=hostVector.getData();
            DistributedGridIO_MPIIOBase<MeshFunctionType>::load(fileName,meshFunction,data);
            meshFunction.getData()=hostVector;
            return true;
         }
#endif
         std::cout << "MPIIO can be used only when MPI is initialized." << std::endl;
         return false;
    };
};

template< typename MeshFunction,
          int Dimension,
          typename MeshReal,
          typename Index >
class DistributedGridIO<
   MeshFunction,
   MpiIO,
   Meshes::Grid< Dimension, MeshReal, Devices::Host, Index >,
   Devices::Host >
{
   public:
      using MeshType = Meshes::Grid< Dimension, MeshReal, Devices::Host, Index >;
      using MeshFunctionType = MeshFunction;

      static bool save(const String& fileName, MeshFunctionType &meshFunction)
      {
#ifdef HAVE_MPI
         if(MPI::isInitialized())//i.e. - isUsed
         {
            typename MeshFunctionType::RealType* data=meshFunction.getData().getData();
            return DistributedGridIO_MPIIOBase<MeshFunctionType>::save(fileName,meshFunction,data);
         }
#endif
         std::cout << "MPIIO can be used only when MPI is initialized." << std::endl;
         return false;
    };

      static bool load(const String& fileName,MeshFunctionType &meshFunction)
      {
#ifdef HAVE_MPI
         if(MPI::isInitialized())//i.e. - isUsed
         {
            typename MeshFunctionType::RealType* data = meshFunction.getData().getData();
            return DistributedGridIO_MPIIOBase<MeshFunctionType>::load(fileName,meshFunction,data);
         }
#endif
         std::cout << "MPIIO can be used only when MPI is initialized." << std::endl;
         return false;
    };
};

      } //namespace DistributedMeshes
   } //namespace Meshes
} //namespace TNL
