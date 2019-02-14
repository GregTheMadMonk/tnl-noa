/***************************************************************************
                          DistributedGridIO_NeshFunction.h  -  description
                             -------------------
    begin                : October 5, 2017
    copyright            : (C) 2017 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/Functions/VectorField.h>
#include <TNL/Meshes/Grid.h>

namespace TNL {
namespace Meshes {   
namespace DistributedMeshes {

template< int Size,
      typename MeshFunction,
      int Dimension,
      typename Real,
      typename Device,
      typename Index > 
class DistributedGridIO< Functions::VectorField< Size, MeshFunction >,
      MpiIO,
      Meshes::Grid< Dimension, Real, Device, Index >,
      Device >
{
    public:
    static bool save(const String& fileName, Functions::VectorField<Size,MeshFunction > &vectorField)
    {
#ifdef HAVE_MPI
        if(Communicators::MpiCommunicator::IsInitialized())//i.e. - isUsed
        {
            auto *distrGrid=vectorField.getMesh().getDistributedMesh();
			if(distrGrid==NULL)
			{
				return vectorField.save(fileName);
			}

            MPI_Comm group=*((MPI_Comm*)(distrGrid->getCommunicationGroup()));

           //write 
           MPI_File file;
           MPI_File_open( group,
                          const_cast< char* >( fileName.getString() ),
                          MPI_MODE_CREATE | MPI_MODE_WRONLY,
                          MPI_INFO_NULL,
                          &file);

          
           int offset=0; //global offset -> every mesh function creates it's own data types we need manage global offset      
           if(Communicators::MpiCommunicator::GetRank(group)==0)
               offset+=writeVectorFieldHeader(file,vectorField);
           MPI_Bcast(&offset, 1, MPI_INT,0, group);
           
           for( int i = 0; i < vectorField.getVectorDimension(); i++ )
           {
               typename MeshFunction::RealType * data=vectorField[i]->getData().getData();  //here manage data transfer Device...
               int size = DistributedGridIO_MPIIOBase<MeshFunction>::save(file,*(vectorField[i]),data,offset);
               offset+=size;
               if( size==0  )
                  return false;
           }

           MPI_File_close(&file); 
           return true;           
        }
#endif
        std::cout << "MPIIO can be used only with MPICommunicator." << std::endl;
        return false;
      
    };

#ifdef HAVE_MPI
	private:
	  static unsigned int writeVectorFieldHeader(MPI_File &file,Functions::VectorField<Size,MeshFunction> &vectorField)
	  {
			unsigned int size=0;
		    int count;
		    MPI_Status wstatus;

		    //Magic
		    MPI_File_write( file, const_cast< void* >( ( const void* ) "TNLMN" ), 5, MPI_CHAR,&wstatus );
		    MPI_Get_count(&wstatus,MPI_CHAR,&count);
		    size+=count*sizeof(char);

		    //VectorField type
		    String vectorFieldSerializationType=vectorField.getSerializationTypeVirtual();
		    int vectorFieldSerializationTypeLength=vectorFieldSerializationType.getLength();
		    MPI_File_write(file,&vectorFieldSerializationTypeLength,1,MPI_INT,&wstatus);
		    MPI_Get_count(&wstatus,MPI_INT,&count);
		    size+=count*sizeof(int);
		    MPI_File_write(file,vectorFieldSerializationType.getString(),vectorFieldSerializationType.getLength(),MPI_CHAR,&wstatus);
		    MPI_Get_count(&wstatus,MPI_CHAR,&count);
		    size+=count*sizeof(char);

		    return size;
	  }

      static unsigned int readVectorFieldHeader(MPI_File &file,Functions::VectorField<Size,MeshFunction> &vectorField)
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

            return size;
      }
#endif

    public:
    static bool load(const String& fileName, Functions::VectorField<Size,MeshFunction> &vectorField)
    {
#ifdef HAVE_MPI
        if(Communicators::MpiCommunicator::IsInitialized())//i.e. - isUsed
        {
            auto *distrGrid=vectorField.getMesh().getDistributedMesh();
			if(distrGrid==NULL)
			{
				return vectorField.save(fileName);
			}

            MPI_Comm group=*((MPI_Comm*)(distrGrid->getCommunicationGroup()));

           //write 
           MPI_File file;
           MPI_File_open( group,
                          const_cast< char* >( fileName.getString() ),
                          MPI_MODE_RDONLY,
                          MPI_INFO_NULL,
                          &file);

          
           int offset=0; //global offset -> every meshfunctoion creates it's own datatypes we need manage global offset      
           if(Communicators::MpiCommunicator::GetRank(group)==0)
               offset+=readVectorFieldHeader(file,vectorField);
           MPI_Bcast(&offset, 1, MPI_INT,0, group);
           
           for( int i = 0; i < vectorField.getVectorDimension(); i++ )
           {
               typename MeshFunction::RealType * data=vectorField[i]->getData().getData();  //here manage data transfer Device...
               int size = DistributedGridIO_MPIIOBase<MeshFunction>::load(file,*(vectorField[i]),data,offset);
               offset+=size;
               if( size==0  )
                  return false;
           }

           MPI_File_close(&file); 
           return true;           
        }
#endif
        std::cout << "MPIIO can be used only with MPICommunicator." << std::endl;
        return false;
      
    };

};

}
}
}
