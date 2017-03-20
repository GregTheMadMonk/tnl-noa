/***************************************************************************
                          tnlFastSweeping2D_CUDA_v2_impl.h  -  description
                             -------------------
    begin                : Oct 15 , 2015
    copyright            : (C) 2015 by Tomas Sobotik
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/
#ifndef TNLFASTSWEEPING2D_IMPL_H_
#define TNLFASTSWEEPING2D_IMPL_H_

#include "tnlFastSweeping.h"


template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
String tnlFastSweeping< tnlGrid< 2,MeshReal, Device, MeshIndex >, Real, Index > :: getType()
{
	   return String( "tnlFastSweeping< " ) +
	          MeshType::getType() + ", " +
	          ::getType< Real >() + ", " +
	          ::getType< Index >() + " >";
}




template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
bool tnlFastSweeping< tnlGrid< 2,MeshReal, Device, MeshIndex >, Real, Index > :: init( const Config::ParameterContainer& parameters )
{
	const String& meshFile = parameters.getParameter< String >( "mesh" );

	if( ! Mesh.load( meshFile ) )
	{
		   cerr << "I am not able to load the mesh from the file " << meshFile << "." << endl;
		   return false;
	}


	const String& initialCondition = parameters.getParameter <String>("initial-condition");
	if( ! dofVector.load( initialCondition ) )
	{
		   cerr << "I am not able to load the initial condition from the file " << meshFile << "." << endl;
		   return false;
	}

	h = Mesh.getSpaceSteps().x();
	counter = 0;

	const String& exact_input = parameters.getParameter< String >( "exact-input" );

	if(exact_input == "no")
		exactInput=false;
	else
		exactInput=true;


#ifdef HAVE_CUDA

	cudaMalloc(&(cudaDofVector), this->dofVector.getSize()*sizeof(double));
	cudaMemcpy(cudaDofVector, this->dofVector.getData(), this->dofVector.getSize()*sizeof(double), cudaMemcpyHostToDevice);

	cudaMalloc(&(cudaDofVector2), this->dofVector.getSize()*sizeof(double));
	cudaMemcpy(cudaDofVector2, this->dofVector.getData(), this->dofVector.getSize()*sizeof(double), cudaMemcpyHostToDevice);


	cudaMalloc(&(this->cudaSolver), sizeof(tnlFastSweeping< tnlGrid< 2,MeshReal, Device, MeshIndex >, Real, Index >));
	cudaMemcpy(this->cudaSolver, this,sizeof(tnlFastSweeping< tnlGrid< 2,MeshReal, Device, MeshIndex >, Real, Index >), cudaMemcpyHostToDevice);

#endif

	int n = Mesh.getDimensions().x();
	dim3 threadsPerBlock(16, 16);
	dim3 numBlocks(n/16 + 1 ,n/16 +1);

	initCUDA<<<numBlocks,threadsPerBlock>>>(this->cudaSolver);
	cudaDeviceSynchronize();
	checkCudaDevice;

	return true;
}





template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
bool tnlFastSweeping< tnlGrid< 2,MeshReal, Device, MeshIndex >, Real, Index > :: run()
{
//
//	for(Index i = 0; i < Mesh.getDimensions().x(); i++)
//	{
//		for(Index j = 0; j < Mesh.getDimensions().y(); j++)
//		{
//			updateValue(i,j);
//		}
//	}
//
///*---------------------------------------------------------------------------------------------------------------------------*/
//
//	for(Index i = Mesh.getDimensions().x() - 1; i > -1; i--)
//	{
//		for(Index j = 0; j < Mesh.getDimensions().y(); j++)
//		{
//			updateValue(i,j);
//		}
//	}
//
///*---------------------------------------------------------------------------------------------------------------------------*/
//
//	for(Index i = Mesh.getDimensions().x() - 1; i > -1; i--)
//	{
//		for(Index j = Mesh.getDimensions().y() - 1; j > -1; j--)
//		{
//			updateValue(i,j);
//		}
//	}
//
///*---------------------------------------------------------------------------------------------------------------------------*/
//	for(Index i = 0; i < Mesh.getDimensions().x(); i++)
//	{
//		for(Index j = Mesh.getDimensions().y() - 1; j > -1; j--)
//		{
//			updateValue(i,j);
//		}
//	}
//
///*---------------------------------------------------------------------------------------------------------------------------*/
//
//
//	dofVector.save("u-00001.tnl");

	int n = Mesh.getDimensions().x();
	dim3 threadsPerBlock(27, 27);
	dim3 numBlocks(1 ,1);

//	for(int i = 2*n - 1; i > -1; i--)
	{
		runCUDA<<<numBlocks,threadsPerBlock>>>(this->cudaSolver,4,0);
		cudaDeviceSynchronize();
	}
	cudaDeviceSynchronize();
	checkCudaDevice;
////	for(int i = 0; i < 2*n ; i++)
//	{
//		runCUDA<<<numBlocks,threadsPerBlock>>>(this->cudaSolver,1,0);
//		cudaDeviceSynchronize();
//	}
//	cudaDeviceSynchronize();
//	checkCudaDevice;
////	for(int i = 0; i < 2*n ; i++)
//	{
//		runCUDA<<<numBlocks,threadsPerBlock>>>(this->cudaSolver,2,0);
//		cudaDeviceSynchronize();
//	}
//	cudaDeviceSynchronize();
//	checkCudaDevice;
////	for(int i = 2*n - 1; i > -1; i--)
//	{
//		runCUDA<<<numBlocks,threadsPerBlock>>>(this->cudaSolver,3,0);
//		cudaDeviceSynchronize();
//	}
//
//	cudaDeviceSynchronize();
//	checkCudaDevice;

	cudaMemcpy(this->dofVector.getData(), cudaDofVector, this->dofVector.getSize()*sizeof(double), cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();
	cudaFree(cudaDofVector);
	cudaFree(cudaDofVector2);
	cudaFree(cudaSolver);
	dofVector.save("u-00001.tnl");
	cudaDeviceSynchronize();
	return true;
}




#ifdef HAVE_CUDA


template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
__device__
void tnlFastSweeping< tnlGrid< 2,MeshReal, Device, MeshIndex >, Real, Index > :: updateValue( Index i, Index j)
{
	Index index = Mesh.getCellIndex(CoordinatesType(i,j));
	Real value = cudaDofVector[index];
	Real a,b, tmp;

	if( i == 0 )
		a = cudaDofVector[Mesh.template getCellNextToCell<1,0>(index)];
	else if( i == Mesh.getDimensions().x() - 1 )
		a = cudaDofVector[Mesh.template getCellNextToCell<-1,0>(index)];
	else
	{
		a = fabsMin( cudaDofVector[Mesh.template getCellNextToCell<-1,0>(index)],
				 cudaDofVector[Mesh.template getCellNextToCell<1,0>(index)] );
	}

	if( j == 0 )
		b = cudaDofVector[Mesh.template getCellNextToCell<0,1>(index)];
	else if( j == Mesh.getDimensions().y() - 1 )
		b = cudaDofVector[Mesh.template getCellNextToCell<0,-1>(index)];
	else
	{
		b = fabsMin( cudaDofVector[Mesh.template getCellNextToCell<0,-1>(index)],
				 cudaDofVector[Mesh.template getCellNextToCell<0,1>(index)] );
	}


	if(abs(a-b) >= h)
		tmp = fabsMin(a,b) + sign(value)*h;
	else
		tmp = 0.5 * (a + b + sign(value)*sqrt(2.0 * h * h - (a - b) * (a - b) ) );

	cudaDofVector[index]  = fabsMin(value, tmp);

}


template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
__device__
bool tnlFastSweeping< tnlGrid< 2,MeshReal, Device, MeshIndex >, Real, Index > :: initGrid()
{
	int gx = threadIdx.x + blockDim.x*blockIdx.x;
	int gy = blockDim.y*blockIdx.y + threadIdx.y;
	int gid = Mesh.getCellIndex(CoordinatesType(gx,gy));

	int total = blockDim.x*gridDim.x;



	Real tmp = 0.0;
	int flag = 0;
	counter = 0;
	tmp = sign(cudaDofVector[Mesh.getCellIndex(CoordinatesType(gx,gy))]);


	if(!exactInput)
	{
		cudaDofVector[gid]=cudaDofVector[gid]=0.5*h*sign(cudaDofVector[gid]);
	}
	__threadfence();
//	printf("-----------------------------------------------------------------------------------\n");

	__threadfence();

	if(gx > 0 && gx < Mesh.getDimensions().x()-1)
	{
		if(gy > 0 && gy < Mesh.getDimensions().y()-1)
		{

			Index j = gy;
			Index i = gx;
//			 tmp = sign(cudaDofVector[Mesh.getCellIndex(CoordinatesType(i,j))]);

			if(tmp == 0.0)
			{}
			else if(cudaDofVector[Mesh.getCellIndex(CoordinatesType(i+1,j))]*tmp < 0.0 ||
					cudaDofVector[Mesh.getCellIndex(CoordinatesType(i-1,j))]*tmp < 0.0 ||
					cudaDofVector[Mesh.getCellIndex(CoordinatesType(i,j+1))]*tmp < 0.0 ||
					cudaDofVector[Mesh.getCellIndex(CoordinatesType(i,j-1))]*tmp < 0.0 )
			{}
			else
				flag=1;//cudaDofVector[Mesh.getCellIndex(CoordinatesType(i,j))] = tmp*INT_MAX;
		}
	}

//	printf("gx: %d, gy: %d, gid: %d \n", gx, gy,gid);
//	printf("****************************************************************\n");
//	printf("gx: %d, gy: %d, gid: %d \n", gx, gy,gid);
	if(gx > 0 && gx < Mesh.getDimensions().x()-1 && gy == 0)
	{
//		printf("gx: %d, gy: %d, gid: %d \n", gx, gy,gid);
		Index j = 0;
		Index i = gx;
//		tmp = sign(cudaDofVector[Mesh.getCellIndex(CoordinatesType(i,j))]);


		if(tmp == 0.0)
		{}
		else if(cudaDofVector[Mesh.getCellIndex(CoordinatesType(i+1,j))]*tmp < 0.0 ||
				cudaDofVector[Mesh.getCellIndex(CoordinatesType(i-1,j))]*tmp < 0.0 ||
				cudaDofVector[Mesh.getCellIndex(CoordinatesType(i,j+1))]*tmp < 0.0 )
		{}
		else
			flag = 1;//cudaDofVector[Mesh.getCellIndex(CoordinatesType(i,j))] = tmp*INT_MAX;
	}

//	printf("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n");
	if(gx > 0 && gx < Mesh.getDimensions().x()-1 && gy == Mesh.getDimensions().y() - 1)
	{
		Index i = gx;
		Index j = Mesh.getDimensions().y() - 1;
//		tmp = sign(cudaDofVector[Mesh.getCellIndex(CoordinatesType(i,j))]);


		if(tmp == 0.0)
		{}
		else if(cudaDofVector[Mesh.getCellIndex(CoordinatesType(i+1,j))]*tmp < 0.0 ||
				cudaDofVector[Mesh.getCellIndex(CoordinatesType(i-1,j))]*tmp < 0.0 ||
				cudaDofVector[Mesh.getCellIndex(CoordinatesType(i,j-1))]*tmp < 0.0 )
		{}
		else
			flag = 1;//cudaDofVector[Mesh.getCellIndex(CoordinatesType(i,j))] = tmp*INT_MAX;
	}

//	printf("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");
	if(gy > 0 && gy < Mesh.getDimensions().y()-1 && gx == 0)
	{
		Index j = gy;
		Index i = 0;
//		tmp = sign(cudaDofVector[Mesh.getCellIndex(CoordinatesType(i,j))]);


		if(tmp == 0.0)
		{}
		else if(cudaDofVector[Mesh.getCellIndex(CoordinatesType(i+1,j))]*tmp < 0.0 ||
				cudaDofVector[Mesh.getCellIndex(CoordinatesType(i,j+1))]*tmp < 0.0 ||
				cudaDofVector[Mesh.getCellIndex(CoordinatesType(i,j-1))]*tmp < 0.0 )
		{}
		else
			flag = 1;//cudaDofVector[Mesh.getCellIndex(CoordinatesType(i,j))] = tmp*INT_MAX;
	}
//	printf("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@\n");
	if(gy > 0 && gy < Mesh.getDimensions().y()-1  && gx == Mesh.getDimensions().x() - 1)
	{
		Index j = gy;
		Index i = Mesh.getDimensions().x() - 1;
//		tmp = sign(cudaDofVector[Mesh.getCellIndex(CoordinatesType(i,j))]);


		if(tmp == 0.0)
		{}
		else if(cudaDofVector[Mesh.getCellIndex(CoordinatesType(i-1,j))]*tmp < 0.0 ||
				cudaDofVector[Mesh.getCellIndex(CoordinatesType(i,j+1))]*tmp < 0.0 ||
				cudaDofVector[Mesh.getCellIndex(CoordinatesType(i,j-1))]*tmp < 0.0 )
		{}
		else
			flag = 1;//cudaDofVector[Mesh.getCellIndex(CoordinatesType(i,j))] = tmp*INT_MAX;
	}

//	printf("##################################################################################################\n");
	if(gx == Mesh.getDimensions().x() - 1 &&
	   gy == Mesh.getDimensions().y() - 1)
	{

//		tmp = sign(cudaDofVector[Mesh.getCellIndex(CoordinatesType(gx,gy))]);
		if(cudaDofVector[Mesh.getCellIndex(CoordinatesType(gx-1,gy))]*tmp > 0.0 &&
				cudaDofVector[Mesh.getCellIndex(CoordinatesType(gx,gy-1))]*tmp > 0.0)

			flag = 1;//cudaDofVector[Mesh.getCellIndex(CoordinatesType(gx,gy))] = tmp*INT_MAX;
	}
	if(gx == Mesh.getDimensions().x() - 1 &&
	   gy == 0)
	{

//		tmp = sign(cudaDofVector[Mesh.getCellIndex(CoordinatesType(gx,gy))]);
		if(cudaDofVector[Mesh.getCellIndex(CoordinatesType(gx-1,gy))]*tmp > 0.0 &&
				cudaDofVector[Mesh.getCellIndex(CoordinatesType(gx,gy+1))]*tmp > 0.0)

			flag = 1;//cudaDofVector[Mesh.getCellIndex(CoordinatesType(gx,gy))] = tmp*INT_MAX;
	}
//	printf("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$\n");
	if(gx == 0 &&
	   gy == Mesh.getDimensions().y() - 1)
	{

//		tmp = sign(cudaDofVector[Mesh.getCellIndex(CoordinatesType(gx,gy))]);
		if(cudaDofVector[Mesh.getCellIndex(CoordinatesType(gx+1,gy))]*tmp > 0.0 &&
				cudaDofVector[Mesh.getCellIndex(CoordinatesType(gx,gy-1))]*tmp > 0.0)

			flag = 1;//cudaDofVector[Mesh.getCellIndex(CoordinatesType(gx,gy))] = tmp*INT_MAX;
	}
	if(gx == 0 &&
	   gy == 0)
	{
//		tmp = sign(cudaDofVector[Mesh.getCellIndex(CoordinatesType(gx,gy))]);
		if(cudaDofVector[Mesh.getCellIndex(CoordinatesType(gx+1,gy))]*tmp > 0.0 &&
				cudaDofVector[Mesh.getCellIndex(CoordinatesType(gx,gy+1))]*tmp > 0.0)

			flag = 1;//cudaDofVector[Mesh.getCellIndex(CoordinatesType(gx,gy))] = tmp*INT_MAX;
	}

	__threadfence();

	if(flag==1)
		cudaDofVector[gid] =  tmp*3;
}


template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
__device__
Real tnlFastSweeping< tnlGrid< 2,MeshReal, Device, MeshIndex >, Real, Index > :: fabsMin( Real x, Real y)
{
	Real fx = abs(x);

	Real tmpMin = Min(fx,abs(y));

	if(tmpMin == fx)
		return x;
	else
		return y;


}



__global__ void runCUDA(tnlFastSweeping< tnlGrid< 2,double, tnlHost, int >, double, int >* solver, int sweep, int k)
{

	//int gx = threadIdx.x;
	//int gy = threadIdx.y;
	int id1,id2;
	int nx = solver->Mesh.getDimensions().x()+ threadIdx.x;
	int ny = solver->Mesh.getDimensions().y()+ threadIdx.y;

	int blockCount = solver->Mesh.getDimensions().x()/blockDim.x + 1;

	for(int gy = threadIdx.y; gy < ny;gy+=blockDim.y)
	{
		for(int gx = threadIdx.x; gx < nx;gx+=blockDim.x)
		{
//			if(solver->Mesh.getDimensions().x() > gx && solver->Mesh.getDimensions().y() > gy && gy > -1&& gx > -1)
			{
				id1 = threadIdx.x+threadIdx.y;

				for(int l = 0; l < 2*blockDim.x - 1; l++)
				{
					if(id1 == l)
					{
						if(solver->Mesh.getDimensions().x() > gx && solver->Mesh.getDimensions().y() > gy /*&& gy > -1&& gx > -1*/)
						solver->updateValue(gx,gy);
					}
					__syncthreads();
				}

			}
			//gx+=blockDim.x;
			//__syncthreads();
		}
		//gx = threadIdx.x;
		//gy+=blockDim.y;
		//__syncthreads();
	}
			/*---------------------------------------------------------------------------------------------------------------------------*/
//	gx = blockDim.x*(blockCount-1) + threadIdx.x;
//	gy = threadIdx.y;
	for(int gy = threadIdx.y; gy < ny;gy+=blockDim.y)
	{
		for(int gx = blockDim.x*(blockCount-1) + threadIdx.x; gx >- 1;gx-=blockDim.x)
		{
//			if(solver->Mesh.getDimensions().x() > gx && solver->Mesh.getDimensions().y() > gy && gy > -1&& gx > -1)
			{
				id2 = (blockDim.x - threadIdx.x - 1) + threadIdx.y;

				for(int l = 0; l < 2*blockDim.x - 1; l++)
				{
					if(id2 == l)
					{
						if(solver->Mesh.getDimensions().x() > gx && solver->Mesh.getDimensions().y() > gy /*&& gy > -1&& gx > -1*/)
						solver->updateValue(gx,gy);
					}
					__syncthreads();
				}
			}
			//gx-=blockDim.x;
			//__syncthreads();
		}
		//gx = blockDim.x*(blockCount-1) + threadIdx.x;
		//gy+=blockDim.y;
		//__syncthreads();
	}
			/*---------------------------------------------------------------------------------------------------------------------------*/
//	gx = blockDim.x*(blockCount-1) + threadIdx.x;
//	gy = blockDim.x*(blockCount-1) + threadIdx.y;
	for(int gy = blockDim.x*(blockCount-1) +threadIdx.y; gy >- 1;gy-=blockDim.y)
	{
		for(int gx = blockDim.x*(blockCount-1) + threadIdx.x; gx >- 1;gx-=blockDim.x)
		{
//			if(solver->Mesh.getDimensions().x() > gx && solver->Mesh.getDimensions().y() > gy && gy > -1&& gx > -1)
			{
				id1 = threadIdx.x+threadIdx.y;

				for(int l = 2*blockDim.x - 2; l > -1; l--)
				{
					if(id1 == l)
					{
						if(solver->Mesh.getDimensions().x() > gx && solver->Mesh.getDimensions().y() > gy /*&& gy > -1&& gx > -1*/)
						solver->updateValue(gx,gy);
					}
					__syncthreads();
				}
			}
			//gx-=blockDim.x;
			//__syncthreads();
		}
		//gx = blockDim.x*(blockCount-1) + threadIdx.x;
		//gy-=blockDim.y;
		//__syncthreads();
	}
			/*---------------------------------------------------------------------------------------------------------------------------*/
	//gx = threadIdx.x;
	//gy = blockDim.x*(blockCount-1) +threadIdx.y;
	for(int gy = blockDim.x*(blockCount-1) +threadIdx.y; gy >- 1;gy-=blockDim.y)
	{
		for(int gx = threadIdx.x; gx < nx;gx+=blockDim.x)
		{
//			if(solver->Mesh.getDimensions().x() > gx && solver->Mesh.getDimensions().y() > gy && gy > -1&& gx > -1)
			{
				id2 = (blockDim.x - threadIdx.x - 1) + threadIdx.y;

				for(int l = 2*blockDim.x - 2; l > -1; l--)
				{
					if(id2 == l)
					{
						if(solver->Mesh.getDimensions().x() > gx && solver->Mesh.getDimensions().y() > gy /*&& gy > -1&& gx > -1*/)
						solver->updateValue(gx,gy);
					}
					__syncthreads();
				}
			}
			//gx+=blockDim.x;
			//__syncthreads();
		}
		//gx = threadIdx.x;
		//gy-=blockDim.y;
		///__syncthreads();
	}
			/*---------------------------------------------------------------------------------------------------------------------------*/





}


__global__ void initCUDA(tnlFastSweeping< tnlGrid< 2,double, tnlHost, int >, double, int >* solver)
{
	int gx = threadIdx.x + blockDim.x*blockIdx.x;
	int gy = blockDim.y*blockIdx.y + threadIdx.y;

	if(solver->Mesh.getDimensions().x() > gx && solver->Mesh.getDimensions().y() > gy)
	{
		solver->initGrid();
	}


}
#endif




#endif /* TNLFASTSWEEPING_IMPL_H_ */
