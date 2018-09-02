#include <stdio.h>
#include <mpi.h>
#include <stdlib.h>

int main(int argc, char **argv)
{

   int localsize = 30;
   int overlap=1;
   int globalsize = localsize+2*overlap;

   double * data;
   //data=(double *)malloc(globalsize*globalsize*sizeof(double));
   cudaMalloc((void **)&data, globalsize*globalsize*sizeof(double));

   MPI_Init(&argc,&argv);

   int rank;
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);
   int size;
   MPI_Comm_size(MPI_COMM_WORLD, &size);

   printf("rank: %d, size %d\n",rank, size);

   int fgsize[2],flsize[2],fstarts[2];
   fgsize[0]=size*localsize;
   fgsize[1]=localsize;
   flsize[0]=localsize;
   flsize[1]=localsize;
   fstarts[0]=rank*localsize;
   fstarts[1]=0;

   MPI_Datatype ftype;

   MPI_Type_create_subarray(2,
        fgsize,flsize,fstarts,
        MPI_ORDER_C,MPI_DOUBLE,&ftype);

   MPI_Type_commit(&ftype);


   int agsize[2],alsize[2],astarts[2];
   agsize[0]=globalsize;
   agsize[1]=globalsize;
   alsize[0]=localsize;
   alsize[1]=localsize;
   astarts[0]=overlap;
   astarts[1]=overlap;

   MPI_Datatype atype;

   MPI_Type_create_subarray(2,
        agsize,alsize,astarts,
        MPI_ORDER_C,MPI_DOUBLE,&atype);

   MPI_Type_commit(&atype);


   MPI_File file;
   MPI_File_open(MPI_COMM_WORLD,"./pokus.file",
	MPI_MODE_CREATE|MPI_MODE_WRONLY,
        MPI_INFO_NULL, &file);

   MPI_File_set_view(file,0,MPI_DOUBLE,ftype,"native",MPI_INFO_NULL);

   MPI_Status wstatus;
   MPI_File_write(file,data,1,atype,&wstatus);

   MPI_File_close(&file);

   MPI_Finalize();

   free(data);

return 0;
}

