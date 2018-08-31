/***************************************************************************
                          tnlFastSweeping2D_CUDA_v3_impl.h  -  description
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




__device__ double atomicSet(double* address, double val)
{
	unsigned long long int* address_as_ull =
						  (unsigned long long int*)address;
	unsigned long long int old = *address_as_ull, assumed;
	do {
		assumed = old;
			old = atomicCAS(address_as_ull, assumed,__double_as_longlong(val ));
	} while (assumed != old);
	return __longlong_as_double(old);
}


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
		  std::cerr << "I am not able to load the mesh from the file " << meshFile << "." <<std::endl;
		   return false;
	}


	const String& initialCondition = parameters.getParameter <String>("initial-condition");
	if( ! dofVector.load( initialCondition ) )
	{
		  std::cerr << "I am not able to load the initial condition from the file " << meshFile << "." <<std::endl;
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
	TNL_CHECK_CUDA_DEVICE;

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
	dim3 threadsPerBlock(16, 16);
	dim3 numBlocks(n/16 +1 ,n/16 +1);
	int m =n/16 +1;

	for(int i = 0; i < 2*m -1; i++)
	{
		runCUDA<15><<<numBlocks,threadsPerBlock>>>(this->cudaSolver,1,i);
		//cudaDeviceSynchronize();
	}
//	cudaDeviceSynchronize();
//	TNL_CHECK_CUDA_DEVICE;
//	for(int i = 0; i < 2*m -1; i++)
//	{
//		runCUDA<2><<<numBlocks,threadsPerBlock>>>(this->cudaSolver,2,i);
//		cudaDeviceSynchronize();
//	}
//	cudaDeviceSynchronize();
//	TNL_CHECK_CUDA_DEVICE;
//	for(int i = 0; i < 2*m -1; i++)
//	{
//		runCUDA<4><<<numBlocks,threadsPerBlock>>>(this->cudaSolver,4,i);
//		cudaDeviceSynchronize();
//	}
//	cudaDeviceSynchronize();
//	TNL_CHECK_CUDA_DEVICE;
//	for(int i = 0; i < 2*m -1; i++)
//	{
//		runCUDA<8><<<numBlocks,threadsPerBlock>>>(this->cudaSolver,8,i);
//		cudaDeviceSynchronize();
//	}




//	for(int i = 0; i < (2*m -1)/4 -1; i++)
//	{
//		runCUDA<15><<<numBlocks,threadsPerBlock>>>(this->cudaSolver,15,i);//all
//		cudaDeviceSynchronize();
//	}
//	for(int i = (2*m -1)/4 -1; i < (2*m -1)/2 -1; i++)
//	{
//		runCUDA<5><<<numBlocks,threadsPerBlock>>>(this->cudaSolver,5,i); //two
//		cudaDeviceSynchronize();
//		runCUDA<10><<<numBlocks,threadsPerBlock>>>(this->cudaSolver,10,i); //two
//		cudaDeviceSynchronize();
//	}
//	for(int i = (2*m -1)/2 -1; i < (2*m -1)/2 +1; i++)
//	{
//		runCUDA<1><<<numBlocks,threadsPerBlock>>>(this->cudaSolver,1,i); //separate
//		cudaDeviceSynchronize();
//		runCUDA<2><<<numBlocks,threadsPerBlock>>>(this->cudaSolver,2,i); //separate
//		cudaDeviceSynchronize();
//		runCUDA<4><<<numBlocks,threadsPerBlock>>>(this->cudaSolver,4,i); //separate
//		cudaDeviceSynchronize();
//		runCUDA<8><<<numBlocks,threadsPerBlock>>>(this->cudaSolver,8,i); //separate
//		cudaDeviceSynchronize();
//	}
//	for(int i = (2*m -1)/2 +1; i < (2*m -1/4)*3 +1; i++)
//	{
//		runCUDA<5><<<numBlocks,threadsPerBlock>>>(this->cudaSolver,5,i); //two
//		cudaDeviceSynchronize();
//		runCUDA<10><<<numBlocks,threadsPerBlock>>>(this->cudaSolver,10,i); //two
//		cudaDeviceSynchronize();
//	}
//	for(int i = (2*m -1/4)*3 +1; i < 2*m -1; i++)
//	{
//		runCUDA<15><<<numBlocks,threadsPerBlock>>>(this->cudaSolver,15,i);//all
//		cudaDeviceSynchronize();
//	}
cudaDeviceSynchronize();
	TNL_CHECK_CUDA_DEVICE;

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

	atomicSet(&cudaDofVector[index],fabsMin(value, tmp));

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
//	Real fx = abs(x);
//
//	Real tmpMin = Min(fx,abs(y));

	if(abs(y) > abs(x))
		return x;
	else
		return y;


}


template<>
__global__ void runCUDA<1>(tnlFastSweeping< tnlGrid< 2,double, TNL::Devices::Host, int >, double, int >* solver, int sweep, int k)
{

	if(blockIdx.x+blockIdx.y == k)
	{
		int gx = threadIdx.x + blockDim.x*blockIdx.x;
		int gy = threadIdx.y + blockDim.y*blockIdx.y;

		int id1 = threadIdx.x+threadIdx.y;

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
			/*---------------------------------------------------------------------------------------------------------------------------*/
}
	template<>
	__global__ void runCUDA<2>(tnlFastSweeping< tnlGrid< 2,double, TNL::Devices::Host, int >, double, int >* solver, int sweep, int k)
	{
	if((gridDim.x - blockIdx.x - 1)+blockIdx.y == k)
	{
		int gx = threadIdx.x + blockDim.x*blockIdx.x;
		int gy = threadIdx.y + blockDim.y*blockIdx.y;

		int id2 = (blockDim.x - threadIdx.x - 1) + threadIdx.y;

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
	}			/*---------------------------------------------------------------------------------------------------------------------------*/
	template<>
	__global__ void runCUDA<4>(tnlFastSweeping< tnlGrid< 2,double, TNL::Devices::Host, int >, double, int >* solver, int sweep, int k)
	{
	if(blockIdx.x+blockIdx.y == gridDim.x+gridDim.y-k-2)
		{
		int gx = threadIdx.x + blockDim.x*blockIdx.x;
		int gy = threadIdx.y + blockDim.y*blockIdx.y;

		int id1 = threadIdx.x+threadIdx.y;

				for(int l = 2*blockDim.x - 2; l > -1; l--)
				{
					if(id1 == l)
					{
						if(solver->Mesh.getDimensions().x() > gx && solver->Mesh.getDimensions().y() > gy /*&& gy > -1&& gx > -1*/)
						solver->updateValue(gx,gy);
						return;
					}
					__syncthreads();
				}

		}
			/*---------------------------------------------------------------------------------------------------------------------------*/

	}

	template<>
	__global__ void runCUDA<8>(tnlFastSweeping< tnlGrid< 2,double, TNL::Devices::Host, int >, double, int >* solver, int sweep, int k)
	{
	if((gridDim.x - blockIdx.x - 1)+blockIdx.y == gridDim.x+gridDim.y-k-2)
		{
		int gx = threadIdx.x + blockDim.x*blockIdx.x;
		int gy = threadIdx.y + blockDim.y*blockIdx.y;

		int id2 = (blockDim.x - threadIdx.x - 1) + threadIdx.y;

				for(int l = 2*blockDim.x - 2; l > -1; l--)
				{
					if(id2 == l)
					{
						if(solver->Mesh.getDimensions().x() > gx && solver->Mesh.getDimensions().y() > gy /*&& gy > -1&& gx > -1*/)
						solver->updateValue(gx,gy);
						return;
					}
					__syncthreads();
				}

		}
			/*---------------------------------------------------------------------------------------------------------------------------*/





}


	template<>
		__global__ void runCUDA<5>(tnlFastSweeping< tnlGrid< 2,double, TNL::Devices::Host, int >, double, int >* solver, int sweep, int k)
		{

			if(blockIdx.x+blockIdx.y == k)
			{
				int gx = threadIdx.x + blockDim.x*blockIdx.x;
				int gy = threadIdx.y + blockDim.y*blockIdx.y;

				int id1 = threadIdx.x+threadIdx.y;

								for(int l = 0; l < 2*blockDim.x - 1; l++)
								{
									if(id1 == l)
									{
										if(solver->Mesh.getDimensions().x() > gx && solver->Mesh.getDimensions().y() > gy /*&& gy > -1&& gx > -1*/)
										solver->updateValue(gx,gy);
										return;
									}
									__syncthreads();
								}

			}
			else if(blockIdx.x+blockIdx.y == gridDim.x+gridDim.y-k-2)
				{
				int gx = threadIdx.x + blockDim.x*blockIdx.x;
				int gy = threadIdx.y + blockDim.y*blockIdx.y;

				int id1 = threadIdx.x+threadIdx.y;

						for(int l = 2*blockDim.x - 2; l > -1; l--)
						{
							if(id1 == l)
							{
								if(solver->Mesh.getDimensions().x() > gx && solver->Mesh.getDimensions().y() > gy /*&& gy > -1&& gx > -1*/)
								solver->updateValue(gx,gy);
								return;
							}
							__syncthreads();
						}

				}
		}


	template<>
		__global__ void runCUDA<10>(tnlFastSweeping< tnlGrid< 2,double, TNL::Devices::Host, int >, double, int >* solver, int sweep, int k)
		{
			if((gridDim.x - blockIdx.x - 1)+blockIdx.y == k)
			{
				int gx = threadIdx.x + blockDim.x*blockIdx.x;
				int gy = threadIdx.y + blockDim.y*blockIdx.y;

				int id2 = (blockDim.x - threadIdx.x - 1) + threadIdx.y;

						for(int l = 0; l < 2*blockDim.x - 1; l++)
						{
							if(id2 == l)
							{
								if(solver->Mesh.getDimensions().x() > gx && solver->Mesh.getDimensions().y() > gy /*&& gy > -1&& gx > -1*/)
								solver->updateValue(gx,gy);
								return;
							}
							__syncthreads();
						}

			}

			else if((gridDim.x - blockIdx.x - 1)+blockIdx.y == gridDim.x+gridDim.y-k-2)
				{
				int gx = threadIdx.x + blockDim.x*blockIdx.x;
				int gy = threadIdx.y + blockDim.y*blockIdx.y;

				int id2 = (blockDim.x - threadIdx.x - 1) + threadIdx.y;

						for(int l = 2*blockDim.x - 2; l > -1; l--)
						{
							if(id2 == l)
							{
								if(solver->Mesh.getDimensions().x() > gx && solver->Mesh.getDimensions().y() > gy /*&& gy > -1&& gx > -1*/)
								solver->updateValue(gx,gy);
								return;
							}
							__syncthreads();
						}

				}

		}



	template<>
	__global__ void runCUDA<15>(tnlFastSweeping< tnlGrid< 2,double, TNL::Devices::Host, int >, double, int >* solver, int sweep, int k)
	{

		if(blockIdx.x+blockIdx.y == k)
		{
			int gx = threadIdx.x + blockDim.x*blockIdx.x;
			int gy = threadIdx.y + blockDim.y*blockIdx.y;

			int id1 = threadIdx.x+threadIdx.y;

							for(int l = 0; l < 2*blockDim.x - 1; l++)
							{
								if(id1 == l)
								{
									if(solver->Mesh.getDimensions().x() > gx && solver->Mesh.getDimensions().y() > gy /*&& gy > -1&& gx > -1*/)
									solver->updateValue(gx,gy);
									return;
								}
								__syncthreads();
							}

		}
				/*---------------------------------------------------------------------------------------------------------------------------*/

		if((gridDim.x - blockIdx.x - 1)+blockIdx.y == k)
		{
			int gx = threadIdx.x + blockDim.x*blockIdx.x;
			int gy = threadIdx.y + blockDim.y*blockIdx.y;

			int id2 = (blockDim.x - threadIdx.x - 1) + threadIdx.y;

					for(int l = 0; l < 2*blockDim.x - 1; l++)
					{
						if(id2 == l)
						{
							if(solver->Mesh.getDimensions().x() > gx && solver->Mesh.getDimensions().y() > gy /*&& gy > -1&& gx > -1*/)
							solver->updateValue(gx,gy);
							return;
						}
						__syncthreads();
					}

		}
				/*---------------------------------------------------------------------------------------------------------------------------*/

		if(blockIdx.x+blockIdx.y == gridDim.x+gridDim.y-k-2)
			{
			int gx = threadIdx.x + blockDim.x*blockIdx.x;
			int gy = threadIdx.y + blockDim.y*blockIdx.y;

			int id1 = threadIdx.x+threadIdx.y;

					for(int l = 2*blockDim.x - 2; l > -1; l--)
					{
						if(id1 == l)
						{
							if(solver->Mesh.getDimensions().x() > gx && solver->Mesh.getDimensions().y() > gy /*&& gy > -1&& gx > -1*/)
							solver->updateValue(gx,gy);
							return;
						}
						__syncthreads();
					}

			}
				/*---------------------------------------------------------------------------------------------------------------------------*/

		if((gridDim.x - blockIdx.x - 1)+blockIdx.y == gridDim.x+gridDim.y-k-2)
			{
			int gx = threadIdx.x + blockDim.x*blockIdx.x;
			int gy = threadIdx.y + blockDim.y*blockIdx.y;

			int id2 = (blockDim.x - threadIdx.x - 1) + threadIdx.y;

					for(int l = 2*blockDim.x - 2; l > -1; l--)
					{
						if(id2 == l)
						{
							if(solver->Mesh.getDimensions().x() > gx && solver->Mesh.getDimensions().y() > gy /*&& gy > -1&& gx > -1*/)
							solver->updateValue(gx,gy);
							return;
						}
						__syncthreads();
					}

			}
				/*---------------------------------------------------------------------------------------------------------------------------*/





	}



























__global__ void initCUDA(tnlFastSweeping< tnlGrid< 2,double, TNL::Devices::Host, int >, double, int >* solver)
{
	int gx = threadIdx.x + blockDim.x*blockIdx.x;
	int gy = blockDim.y*blockIdx.y + threadIdx.y;

	if(solver->Mesh.getDimensions().x() > gx && solver->Mesh.getDimensions().y() > gy)
	{
		solver->initGrid();
	}


}
#endif






//__global__ void runCUDA(tnlFastSweeping< tnlGrid< 2,double, TNL::Devices::Host, int >, double, int >* solver, int sweep, int k)
//{
//
//	if(sweep==1 && blockIdx.x+blockIdx.y == k)
//	{
//		int gx = threadIdx.x + blockDim.x*blockIdx.x;
//		int gy = threadIdx.y + blockDim.y*blockIdx.y;
//
//		int id1 = threadIdx.x+threadIdx.y;
//
//						for(int l = 0; l < 2*blockDim.x - 1; l++)
//						{
//							if(id1 == l)
//							{
//								if(solver->Mesh.getDimensions().x() > gx && solver->Mesh.getDimensions().y() > gy /*&& gy > -1&& gx > -1*/)
//								solver->updateValue(gx,gy);
//							}
//							__syncthreads();
//						}
//
//	}
//			/*---------------------------------------------------------------------------------------------------------------------------*/
//
//	else if(sweep==2 && (gridDim.x - blockIdx.x - 1)+blockIdx.y == k)
//	{
//		int gx = threadIdx.x + blockDim.x*blockIdx.x;
//		int gy = threadIdx.y + blockDim.y*blockIdx.y;
//
//		int id2 = (blockDim.x - threadIdx.x - 1) + threadIdx.y;
//
//				for(int l = 0; l < 2*blockDim.x - 1; l++)
//				{
//					if(id2 == l)
//					{
//						if(solver->Mesh.getDimensions().x() > gx && solver->Mesh.getDimensions().y() > gy /*&& gy > -1&& gx > -1*/)
//						solver->updateValue(gx,gy);
//					}
//					__syncthreads();
//				}
//
//	}
//			/*---------------------------------------------------------------------------------------------------------------------------*/
//
//	else if(sweep==4 && blockIdx.x+blockIdx.y == gridDim.x+gridDim.y-k-2)
//		{
//		int gx = threadIdx.x + blockDim.x*blockIdx.x;
//		int gy = threadIdx.y + blockDim.y*blockIdx.y;
//
//		int id1 = threadIdx.x+threadIdx.y;
//
//				for(int l = 2*blockDim.x - 2; l > -1; l--)
//				{
//					if(id1 == l)
//					{
//						if(solver->Mesh.getDimensions().x() > gx && solver->Mesh.getDimensions().y() > gy /*&& gy > -1&& gx > -1*/)
//						solver->updateValue(gx,gy);
//						return;
//					}
//					__syncthreads();
//				}
//
//		}
//			/*---------------------------------------------------------------------------------------------------------------------------*/
//
//	else if(sweep==8 && (gridDim.x - blockIdx.x - 1)+blockIdx.y == gridDim.x+gridDim.y-k-2)
//		{
//		int gx = threadIdx.x + blockDim.x*blockIdx.x;
//		int gy = threadIdx.y + blockDim.y*blockIdx.y;
//
//		int id2 = (blockDim.x - threadIdx.x - 1) + threadIdx.y;
//
//				for(int l = 2*blockDim.x - 2; l > -1; l--)
//				{
//					if(id2 == l)
//					{
//						if(solver->Mesh.getDimensions().x() > gx && solver->Mesh.getDimensions().y() > gy /*&& gy > -1&& gx > -1*/)
//						solver->updateValue(gx,gy);
//						return;
//					}
//					__syncthreads();
//				}
//
//		}
//			/*---------------------------------------------------------------------------------------------------------------------------*/
//
//
//
//
//
//}


#endif /* TNLFASTSWEEPING_IMPL_H_ */
