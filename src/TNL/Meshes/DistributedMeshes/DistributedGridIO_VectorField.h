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

namespace TNL {
namespace Meshes {   
namespace DistributedMeshes {

//VCT field
template<
        int Size,
        typename MeshFunctionType,
        typename Device> 
class DistributedGridIO<Functions::VectorField<Size,MeshFunctionType>,MpiIO,Device>
{
    public:
    static bool save(const String& fileName, Functions::VectorField<Size,MeshFunctionType> &vectorField)
    {
#ifdef HAVE_MPI
        std::cout << "saving VCT filed" << endl;

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

          
           int offset=0; //global offset -> every meshfunctoion creates it's own datatypes we need manage global offset      
           if(Communicators::MpiCommunicator::GetRank(group)==0)
               offset+=writeVectorFieldHeader(file,vectorField);
           MPI_Bcast(&offset, 1, MPI_INT,0, group);
           
           for( int i = 0; i < vectorField.getVectorDimension(); i++ )
           {
               typename MeshFunctionType::RealType * data=vectorField[i]->getData().getData();  //here manage data transfer Device...
               int size = DistributedGridIO_MPIIOBase<MeshFunctionType>::save(file,*(vectorField[i]),data,offset);
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

	private:
	  static unsigned int writeVectorFieldHeader(MPI_File &file,Functions::VectorField<Size,MeshFunctionType> &vectorField)
	  {
			unsigned int size=0;
		    int count;
		    MPI_Status wstatus;

		    //Magic
		    MPI_File_write( file, const_cast< void* >( ( const void* ) "TNLMN" ), 5, MPI_CHAR,&wstatus );
		    MPI_Get_count(&wstatus,MPI_CHAR,&count);
		    size+=count*sizeof(char);

		    //Meshfunction type
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

/*    static bool load(const String& fileName,MeshFunctionType &meshFunction) 
    {
#ifdef HAVE_MPI
        if(Communicators::MpiCommunicator::IsInitialized())//i.e. - isUsed
        {
            using HostVectorType = Containers::Vector<typename MeshFunctionType::RealType, Devices::Host, typename MeshFunctionType::IndexType >; 
            HostVectorType hostVector;
            hostVector.setLike(meshFunction.getData());
            double * data=hostVector.getData();
            DistributedGridIO_MPIIOBase<MeshFunctionType>::load(fileName,meshFunction,data);
            meshFunction.getData()=hostVector;
            return true;
        }
#endif
        std::cout << "MPIIO can be used only with MPICommunicator." << std::endl;
        return false;
    };

#ifdef HAVE MPI
    private:
        static unsigned int writeVectorFieldHeader(MPI_File &file,Functions::VectorField<Size,MeshFunctionType> &vectorField )
        {
            unsigned int size=0;
            int count;
            MPI_Status wstatus;

            //Magic
            MPI_File_write( file, const_cast< void* >( ( const void* ) "TNLMN" ), 5, MPI_CHAR,&wstatus );
            MPI_Get_count(&wstatus,MPI_CHAR,&count);
            size+=count*sizeof(char);

            //Meshfunction type
            String vectorFieldSerializationType=vectorField.getSerializationTypeVirtual();
            int vectorFieldSerializationTypeLength=vectorFieldSerializationType.getLength();
            MPI_File_write(file,&vectorFieldSerializationTypeLength,1,MPI_INT,&wstatus);
            MPI_Get_count(&wstatus,MPI_INT,&count);
            size+=count*sizeof(int);
            MPI_File_write(file,vectorFieldSerializationType.getString(),vectorFieldSerializationType.getLength(),MPI_CHAR,&wstatus);
            MPI_Get_count(&wstatus,MPI_CHAR,&count);
            size+=count*sizeof(char);

            return size;

        };
#endif
*/

};

}
}
}
