/***************************************************************************
                          tnlNarrowBand2D_CUDA_v4_impl.h  -  description
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
#ifndef TNLNARROWBAND2D_IMPL_H_
#define TNLNARROWBAND2D_IMPL_H_

#define NARROWBAND_SUBGRID_SIZE 32

#include "tnlNarrowBand.h"

__device__
double fabsMin( double x, double y)
{
	double fx = abs(x);

	if(Min(fx,abs(y)) == fx)
		return x;
	else
		return y;
}

__device__
double atomicFabsMin(double* address, double val)
{
	unsigned long long int* address_as_ull =
						  (unsigned long long int*)address;
	unsigned long long int old = *address_as_ull, assumed;
	do {
		assumed = old;
			old = atomicCAS(address_as_ull, assumed,__double_as_longlong( fabsMin(__longlong_as_double(assumed),val) ));
	} while (assumed != old);
	return __longlong_as_double(old);
}


template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
#ifdef HAVE_CUDA
   __device__ __host__
#endif
Real tnlNarrowBand< tnlGrid< 2,MeshReal, Device, MeshIndex >, Real, Index >:: positivePart(const Real arg) const
{
	if(arg > 0.0)
		return arg;
	return 0.0;
}


template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
#ifdef HAVE_CUDA
   __device__ __host__
#endif
Real  tnlNarrowBand< tnlGrid< 2,MeshReal, Device, MeshIndex >, Real, Index > :: negativePart(const Real arg) const
{
	if(arg < 0.0)
		return -arg;
	return 0.0;
}

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
String tnlNarrowBand< tnlGrid< 2,MeshReal, Device, MeshIndex >, Real, Index > :: getType()
{
	   return String( "tnlNarrowBand< " ) +
	          MeshType::getType() + ", " +
	          ::getType< Real >() + ", " +
	          ::getType< Index >() + " >";
}

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
tnlNarrowBand< tnlGrid< 2,MeshReal, Device, MeshIndex >, Real, Index > :: tnlNarrowBand()
:dofVector(Mesh)
{
}


template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
bool tnlNarrowBand< tnlGrid< 2,MeshReal, Device, MeshIndex >, Real, Index > :: init( const Config::ParameterContainer& parameters )
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

	h = Mesh.template getSpaceStepsProducts< 1, 0 >();
	//Entity.refresh();
	counter = 0;

	const String& exact_input = parameters.getParameter< String >( "exact-input" );

	if(exact_input == "no")
		exactInput=false;
	else
		exactInput=true;

	tau = parameters.getParameter< double >( "tau" );

	finalTime = parameters.getParameter< double >( "final-time" );

	statusGridSize = ((Mesh.getDimensions().x() + NARROWBAND_SUBGRID_SIZE-1 ) / NARROWBAND_SUBGRID_SIZE);
#ifdef HAVE_CUDA

	cudaMalloc(&(cudaDofVector), this->dofVector.getData().getSize()*sizeof(double));
	cudaMemcpy(cudaDofVector, this->dofVector.getData().getData(), this->dofVector.getData().getSize()*sizeof(double), cudaMemcpyHostToDevice);

	cudaMalloc(&(cudaDofVector2), this->dofVector.getData().getSize()*sizeof(double));
	cudaMemcpy(cudaDofVector2, this->dofVector.getData().getData(), this->dofVector.getData().getSize()*sizeof(double), cudaMemcpyHostToDevice);

	cudaMalloc(&(cudaStatusVector),  statusGridSize*statusGridSize*sizeof(int));
//	cudaMemcpy(cudaDofVector, this->dofVector.getData().getData(),  statusGridSize*statusGridSize* sizeof(int)), cudaMemcpyHostToDevice);

	cudaMalloc(&reinitialize, sizeof(int));


	cudaMalloc(&(this->cudaSolver), sizeof(tnlNarrowBand< tnlGrid< 2,MeshReal, Device, MeshIndex >, Real, Index >));
	cudaMemcpy(this->cudaSolver, this,sizeof(tnlNarrowBand< tnlGrid< 2,MeshReal, Device, MeshIndex >, Real, Index >), cudaMemcpyHostToDevice);


	cudaDeviceSynchronize();
	TNL_CHECK_CUDA_DEVICE;
#endif

	int n = Mesh.getDimensions().x();

	dim3 threadsPerBlock2(NARROWBAND_SUBGRID_SIZE, NARROWBAND_SUBGRID_SIZE);
	dim3 numBlocks2(statusGridSize ,statusGridSize);
	initSetupGridCUDA<<<numBlocks2,threadsPerBlock2>>>(this->cudaSolver);
	cudaDeviceSynchronize();
	TNL_CHECK_CUDA_DEVICE;
	initSetupGrid2CUDA<<<numBlocks2,1>>>(this->cudaSolver);
	cudaDeviceSynchronize();
	TNL_CHECK_CUDA_DEVICE;


	/*dim3 threadsPerBlock(16, 16);
	dim3 numBlocks(n/16 + 1 ,n/16 +1);*/
	initCUDA<<<numBlocks2,threadsPerBlock2>>>(this->cudaSolver);
	cudaDeviceSynchronize();
	TNL_CHECK_CUDA_DEVICE;


	cout << "Solver initialized." <<std::endl;
	return true;
}





template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
bool tnlNarrowBand< tnlGrid< 2,MeshReal, Device, MeshIndex >, Real, Index > :: run()
{

	int n = Mesh.getDimensions().x();
	dim3 threadsPerBlockFS(1, 512);
	dim3 numBlocksFS(4,1);
	dim3 threadsPerBlockNB(NARROWBAND_SUBGRID_SIZE, NARROWBAND_SUBGRID_SIZE);
	dim3 numBlocksNB(n/NARROWBAND_SUBGRID_SIZE + 1,n/NARROWBAND_SUBGRID_SIZE + 1);

	double time = 0.0;
	int reinit = 0;

	cout << "Hi!" <<std::endl;
	runCUDA<<<numBlocksFS,threadsPerBlockFS>>>(this->cudaSolver,0,0);
	cudaDeviceSynchronize();
	TNL_CHECK_CUDA_DEVICE;
	cout << "Hi2!" <<std::endl;
	while(time < finalTime)
	{
		if(tau+time > finalTime)
			tau=finalTime-time;

		runNarrowBandCUDA<<<numBlocksNB,threadsPerBlockNB>>>(this->cudaSolver,tau);
		cudaDeviceSynchronize();
		TNL_CHECK_CUDA_DEVICE;

		time += tau;


		cudaMemcpy(&reinit, this->reinitialize, sizeof(int), cudaMemcpyDeviceToHost);
		cudaDeviceSynchronize();
		TNL_CHECK_CUDA_DEVICE;
		if(reinit != 0 /*&& time != finalTime */)
		{
			cout << time <<std::endl;

			initSetupGridCUDA<<<numBlocksNB,threadsPerBlockNB>>>(this->cudaSolver);
			cudaDeviceSynchronize();
			TNL_CHECK_CUDA_DEVICE;
			initSetupGrid2CUDA<<<numBlocksNB,1>>>(this->cudaSolver);
			cudaDeviceSynchronize();
			TNL_CHECK_CUDA_DEVICE;
			initCUDA<<<numBlocksNB,threadsPerBlockNB>>>(this->cudaSolver);
			cudaDeviceSynchronize();
			TNL_CHECK_CUDA_DEVICE;
			runCUDA<<<numBlocksFS,threadsPerBlockFS>>>(this->cudaSolver,0,0);
			cudaDeviceSynchronize();
			TNL_CHECK_CUDA_DEVICE;
		}
	}

	//data.setLike(dofVector.getData());
	//cudaMemcpy(data.getData(), cudaDofVector2, this->dofVector.getData().getSize()*sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpy(dofVector.getData().getData(), cudaDofVector2, this->dofVector.getData().getSize()*sizeof(double), cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();
	cudaFree(cudaDofVector);
	cudaFree(cudaDofVector2);
	cudaFree(cudaSolver);
	//data.save("u-00001.tnl");
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
void tnlNarrowBand< tnlGrid< 2,MeshReal, Device, MeshIndex >, Real, Index > :: updateValue( Index i, Index j)
{
	//			1 - with curve,  	2 - to the north of curve, 	4  - to the south of curve,
	//								8 - to the east of curve, 	16 - to the west of curve.
	int subgridID = i/NARROWBAND_SUBGRID_SIZE + (j/NARROWBAND_SUBGRID_SIZE) * ((Mesh.getDimensions().x() + NARROWBAND_SUBGRID_SIZE-1 ) / NARROWBAND_SUBGRID_SIZE);
	if(/*cudaStatusVector[subgridID] != 0 &&*/ i<Mesh.getDimensions().x() && Mesh.getDimensions().y())
	{
		tnlGridEntity< tnlGrid< 2,double, TNL::Devices::Host, int >, 2, tnlGridEntityNoStencilStorage > Entity(Mesh);
		Entity.setCoordinates(CoordinatesType(i,j));
		Entity.refresh();
		tnlNeighborGridEntityGetter<tnlGridEntity< MeshType, 2, tnlGridEntityNoStencilStorage >,2> neighborEntities(Entity);
		Real value = cudaDofVector2[Entity.getIndex()];
		Real a,b, tmp;

		if( i == 0 /*|| (i/NARROWBAND_SUBGRID_SIZE == 0 && !(cudaStatusVector[subgridID] & 9)) */)
			a = cudaDofVector2[neighborEntities.template getEntityIndex< 1,  0 >()];
		else if( i == Mesh.getDimensions().x() - 1 /*|| (i/NARROWBAND_SUBGRID_SIZE == NARROWBAND_SUBGRID_SIZE - 1 && !(cudaStatusVector[subgridID] & 17)) */)
			a = cudaDofVector2[neighborEntities.template getEntityIndex< -1,  0 >()];
		else
		{
			a = fabsMin( cudaDofVector2[neighborEntities.template getEntityIndex< -1,  0 >()],
					 cudaDofVector2[neighborEntities.template getEntityIndex< 1,  0 >()] );
		}

		if( j == 0/* || (j/NARROWBAND_SUBGRID_SIZE == 0 && !(cudaStatusVector[subgridID] & 3)) */)
			b = cudaDofVector2[neighborEntities.template getEntityIndex< 0,  1 >()];
		else if( j == Mesh.getDimensions().y() - 1 /* || (j/NARROWBAND_SUBGRID_SIZE == NARROWBAND_SUBGRID_SIZE - 1 && !(cudaStatusVector[subgridID] & 5))*/ )
			b = cudaDofVector2[neighborEntities.template getEntityIndex< 0,  -1 >()];
		else
		{
			b = fabsMin( cudaDofVector2[neighborEntities.template getEntityIndex< 0,  -1 >()],
					 cudaDofVector2[neighborEntities.template getEntityIndex< 0,  1 >()] );
		}


		if(abs(a-b) >= h)
			tmp = fabsMin(a,b) + sign(value)*h;
		else
			tmp = 0.5 * (a + b + sign(value)*sqrt(2.0 * h * h - (a - b) * (a - b) ) );

	//	cudaDofVector2[Entity.getIndex()]  = fabsMin(value, tmp);
		atomicFabsMin(&(cudaDofVector2[Entity.getIndex()]), tmp);
	}

}


__global__ void initCUDA(tnlNarrowBand< tnlGrid< 2,double, TNL::Devices::Host, int >, double, int >* solver)
{


	int gx = threadIdx.x + blockDim.x*blockIdx.x;
	int gy = blockDim.y*blockIdx.y + threadIdx.y;


	if(solver->Mesh.getDimensions().x() > gx  && solver->Mesh.getDimensions().y() > gy)
	{
		solver->initGrid();
	}


}



template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
__device__
bool tnlNarrowBand< tnlGrid< 2,MeshReal, Device, MeshIndex >, Real, Index > :: initGrid()
{
	int i = threadIdx.x + blockDim.x*blockIdx.x;
	int j = blockDim.y*blockIdx.y + threadIdx.y;

	tnlGridEntity< tnlGrid< 2,double, TNL::Devices::Host, int >, 2, tnlGridEntityNoStencilStorage > Entity(Mesh);

	Entity.setCoordinates(CoordinatesType(i,j));
	Entity.refresh();
	tnlNeighborGridEntityGetter<tnlGridEntity< MeshType, 2, tnlGridEntityNoStencilStorage >,2> neighborEntities(Entity);

	int gid = Entity.getIndex();

	cudaDofVector2[gid] = INT_MAX*sign(cudaDofVector2[gid]);

	if (i >0 && j > 0 && i+1 < Mesh.getDimensions().x() && j+1 < Mesh.getDimensions().y())
	{
		if(cudaDofVector2[gid]*cudaDofVector2[gid+1] <= 0 )
		{
			cudaDofVector2[gid] = sign(cudaDofVector2[gid])*0.5*h;
			cudaDofVector2[gid+1] = sign(cudaDofVector2[gid+1])*0.5*h;
		}
		if( cudaDofVector2[gid]*cudaDofVector2[gid+Mesh.getDimensions().x()] <= 0 )
		{
			cudaDofVector2[gid] = sign(cudaDofVector2[gid])*0.5*h;
			cudaDofVector2[gid+Mesh.getDimensions().x()] = sign(cudaDofVector2[gid+Mesh.getDimensions().x()])*0.5*h;
		}

		if(cudaDofVector2[gid]*cudaDofVector2[gid-1] <= 0 )
		{
			cudaDofVector2[gid] = sign(cudaDofVector2[gid])*0.5*h;
			cudaDofVector2[gid-1] = sign(cudaDofVector2[gid-1])*0.5*h;
		}
		if( cudaDofVector2[gid]*cudaDofVector2[gid-Mesh.getDimensions().x()] <= 0 )
		{
			cudaDofVector2[gid] = sign(cudaDofVector2[gid])*0.5*h;
			cudaDofVector2[gid-Mesh.getDimensions().x()] = sign(cudaDofVector2[gid-Mesh.getDimensions().x()])*0.5*h;
		}
	}


//






//	if(i+1 < Mesh.getDimensions().x() && j+1 < Mesh.getDimensions().y() )
//	{
//		if(cudaDofVector[Entity.getIndex()] > 0)
//		{
//			if(cudaDofVector[neighborEntities.template getEntityIndex< 1,  0 >()] > 0)
//			{
//				if(cudaDofVector[neighborEntities.template getEntityIndex< 0,  1 >()] > 0)
//				{
//					if(cudaDofVector[neighborEntities.template getEntityIndex< 1,  1 >()] > 0)
//						setupSquare1111(i,j);
//					else
//						setupSquare1110(i,j);
//				}
//				else
//				{
//					if(cudaDofVector[neighborEntities.template getEntityIndex< 1,  1 >()] > 0)
//						setupSquare1101(i,j);
//					else
//						setupSquare1100(i,j);
//				}
//			}
//			else
//			{
//				if(cudaDofVector[neighborEntities.template getEntityIndex< 0,  1 >()] > 0)
//				{
//					if(cudaDofVector[neighborEntities.template getEntityIndex< 1,  1 >()] > 0)
//						setupSquare1011(i,j);
//					else
//						setupSquare1010(i,j);
//				}
//				else
//				{
//					if(cudaDofVector[neighborEntities.template getEntityIndex< 1,  1 >()] > 0)
//						setupSquare1001(i,j);
//					else
//						setupSquare1000(i,j);
//				}
//			}
//		}
//		else
//		{
//			if(cudaDofVector[neighborEntities.template getEntityIndex< 1,  0 >()] > 0)
//			{
//				if(cudaDofVector[neighborEntities.template getEntityIndex< 0,  1 >()] > 0)
//				{
//					if(cudaDofVector[neighborEntities.template getEntityIndex< 1,  1 >()] > 0)
//						setupSquare0111(i,j);
//					else
//						setupSquare0110(i,j);
//				}
//				else
//				{
//					if(cudaDofVector[neighborEntities.template getEntityIndex< 1,  1 >()] > 0)
//						setupSquare0101(i,j);
//					else
//						setupSquare0100(i,j);
//				}
//			}
//			else
//			{
//				if(cudaDofVector[neighborEntities.template getEntityIndex< 0,  1 >()] > 0)
//				{
//					if(cudaDofVector[neighborEntities.template getEntityIndex< 1,  1 >()] > 0)
//						setupSquare0011(i,j);
//					else
//						setupSquare0010(i,j);
//				}
//				else
//				{
//					if(cudaDofVector[neighborEntities.template getEntityIndex< 1,  1 >()] > 0)
//						setupSquare0001(i,j);
//					else
//						setupSquare0000(i,j);
//				}
//			}
//		}
//
//	}

	return true;

}


template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
__device__
Real tnlNarrowBand< tnlGrid< 2,MeshReal, Device, MeshIndex >, Real, Index > :: fabsMin( Real x, Real y)
{
	Real fx = abs(x);
	//Real fy = abs(y);

	//Real tmpMin = Min(fx,abs(y));

	if(Min(fx,abs(y)) == fx)
		return x;
	else
		return y;


}



__global__ void runCUDA(tnlNarrowBand< tnlGrid< 2,double, TNL::Devices::Host, int >, double, int >* solver, int sweep, int i)
{


	int gx = 0;
	int gy = threadIdx.y;
	//if(solver->Mesh.getDimensions().x() <= gx || solver->Mesh.getDimensions().y() <= gy)
	//	return;
	int n = solver->Mesh.getDimensions().x();
	int blockCount = n/blockDim.y +1;
	//int gid = solver->Mesh.getDimensions().x() * gy + gx;
	//int max = solver->Mesh.getDimensions().x()*solver->Mesh.getDimensions().x();

	//int id1 = gx+gy;
	//int id2 = (solver->Mesh.getDimensions().x() - gx - 1) + gy;

	if(blockIdx.x==0)
	{
		for(int k = 0; k < n*blockCount + blockDim.y; k++)
		{
			if(threadIdx.y  < k+1 && gy < n)
			{
				solver->updateValue(gx,gy);
				gx++;
				if(gx==n)
				{
					gx=0;
					gy+=blockDim.y;
				}
			}


			__syncthreads();
		}
	}
	else if(blockIdx.x==1)
	{
		gx=n-1;
		gy=threadIdx.y;

		for(int k = 0; k < n*blockCount + blockDim.y; k++)
		{
			if(threadIdx.y  < k+1 && gy < n)
			{
				solver->updateValue(gx,gy);
				gx--;
				if(gx==-1)
				{
					gx=n-1;
					gy+=blockDim.y;
				}
			}


			__syncthreads();
		}
	}
	else if(blockIdx.x==2)
	{
		gx=0;
		gy=n-threadIdx.y-1;
		for(int k = 0; k < n*blockCount + blockDim.y; k++)
		{
			if(threadIdx.y  < k+1 && gy > -1)
			{
				solver->updateValue(gx,gy);
				gx++;
				if(gx==n)
				{
					gx=0;
					gy-=blockDim.y;
				}
			}


			__syncthreads();
		}
	}
	else if(blockIdx.x==3)
	{
		gx=n-1;
		gy=n-threadIdx.y-1;

		for(int k = 0; k < n*blockCount + blockDim.y; k++)
		{
			if(threadIdx.y  < k+1 && gy > -1)
			{
				solver->updateValue(gx,gy);
				gx--;
				if(gx==-1)
				{
					gx=n-1;
					gy-=blockDim.y;
				}
			}


			__syncthreads();
		}
	}

}




__global__ void initSetupGridCUDA(tnlNarrowBand< tnlGrid< 2,double, TNL::Devices::Host, int >, double, int >* solver)
{
	__shared__ double u0;
	int gx = threadIdx.x + blockDim.x*blockIdx.x;
	int gy = blockDim.y*blockIdx.y + threadIdx.y;

	if(solver->Mesh.getDimensions().x() > gx && solver->Mesh.getDimensions().y() > gy)
	{

//		printf("Hello from  block = %d, thread = %d, x = %d, y = %d\n", blockIdx.x + gridDim.x*blockIdx.y,(blockDim.y*blockIdx.y + threadIdx.y)*solver->Mesh.getDimensions().x() + blockDim.x*blockIdx.x + threadIdx.x, threadIdx.x, threadIdx.y);
		if(threadIdx.x+threadIdx.y == 0)
		{
//			printf("Hello from  block = %d, thread = %d, x = %d, y = %d\n", blockIdx.x + gridDim.x*blockIdx.y,(blockDim.y*blockIdx.y + threadIdx.y)*solver->Mesh.getDimensions().x() + blockDim.x*blockIdx.x + threadIdx.x, threadIdx.x, threadIdx.y);

			if(blockIdx.x+blockIdx.y == 0)
				*(solver->reinitialize) = 0;

			solver->cudaStatusVector[blockIdx.x + gridDim.x*blockIdx.y] = 0;

			u0 = solver->cudaDofVector2[(blockDim.y*blockIdx.y + 0)*solver->Mesh.getDimensions().x() + blockDim.x*blockIdx.x + 0];
		}
		__syncthreads();

		double u = solver->cudaDofVector2[(blockDim.y*blockIdx.y + threadIdx.y)*solver->Mesh.getDimensions().x() + blockDim.x*blockIdx.x + threadIdx.x];

		if(u*u0 <=0.0)
			atomicMax(&(solver->cudaStatusVector[blockIdx.x + gridDim.x*blockIdx.y]),1);
	}
//	if(threadIdx.x+threadIdx.y == 0)

//	printf("Bye from  block = %d, thread = %d, x = %d, y = %d\n", blockIdx.x + gridDim.x*blockIdx.y,(blockDim.y*blockIdx.y + threadIdx.y)*solver->Mesh.getDimensions().x() + blockDim.x*blockIdx.x + threadIdx.x, threadIdx.x, threadIdx.y);


}



// run this with one thread per block
__global__ void initSetupGrid2CUDA(tnlNarrowBand< tnlGrid< 2,double, TNL::Devices::Host, int >, double, int >* solver)
{
//	printf("Hello\n");
	if(solver->cudaStatusVector[blockIdx.x + gridDim.x*blockIdx.y] == 1)
	{
//			1 - with curve,  	2 - to the north of curve, 	4  - to the south of curve,
//								8 - to the east of curve, 	16 - to the west of curve.
			if(blockIdx.x > 0)
				atomicAdd(&(solver->cudaStatusVector[blockIdx.x - 1 + gridDim.x*blockIdx.y]), 16);

			if(blockIdx.x < gridDim.x - 1)
				atomicAdd(&(solver->cudaStatusVector[blockIdx.x + 1 + gridDim.x*blockIdx.y]), 8);

			if(blockIdx.y > 0 )
				atomicAdd(&(solver->cudaStatusVector[blockIdx.x + gridDim.x*(blockIdx.y - 1)]), 4);

			if(blockIdx.y < gridDim.y - 1)
				atomicAdd(&(solver->cudaStatusVector[blockIdx.x + gridDim.x*(blockIdx.y + 1)]), 2);
	}


}





__global__ void runNarrowBandCUDA(tnlNarrowBand< tnlGrid< 2,double, TNL::Devices::Host, int >, double, int >* solver, double tau)
{
	int gid = (blockDim.y*blockIdx.y + threadIdx.y)*solver->Mesh.getDimensions().x()+ threadIdx.x;
	int i = threadIdx.x + blockIdx.x*blockDim.x;
	int j = threadIdx.y + blockIdx.y*blockDim.y;

//	if(i+j == 0)
//		printf("Hello\n");

	int blockID = blockIdx.x + blockIdx.y*gridDim.x; /*i/NARROWBAND_SUBGRID_SIZE + (j/NARROWBAND_SUBGRID_SIZE) * ((Mesh.getDimensions().x() + NARROWBAND_SUBGRID_SIZE-1 ) / NARROWBAND_SUBGRID_SIZE);*/

	int status = solver->cudaStatusVector[blockID];

	if(solver->Mesh.getDimensions().x() > i && solver->Mesh.getDimensions().y() > j)
	{

//		if(status != 0)
		{
			tnlGridEntity< tnlGrid< 2,double, TNL::Devices::Host, int >, 2, tnlGridEntityNoStencilStorage > Entity(solver->Mesh);
			Entity.setCoordinates(Containers::StaticVector<2,double>(i,j));
			Entity.refresh();
			tnlNeighborGridEntityGetter<tnlGridEntity< tnlGrid< 2,double, TNL::Devices::Host, int >, 2, tnlGridEntityNoStencilStorage >,2> neighborEntities(Entity);
			double value = solver->cudaDofVector2[Entity.getIndex()];
			double xf,xb,yf,yb, grad, fu, a,b;
			a = b = 0.0;

			if( i == 0 /*|| (threadIdx.x == 0 && !(status & 9)) */)
			{
				xb = value - solver->cudaDofVector2[neighborEntities.template getEntityIndex< 1,  0 >()];
				xf = solver->cudaDofVector2[neighborEntities.template getEntityIndex< 1,  0 >()] - value;
			}
			else if( i == solver->Mesh.getDimensions().x() - 1 /*|| (threadIdx.x == blockDim.x - 1 && !(status & 17)) */)
			{
				xb = value - solver->cudaDofVector2[neighborEntities.template getEntityIndex< -1,  0 >()];
				xf = solver->cudaDofVector2[neighborEntities.template getEntityIndex< -1,  0 >()] - value;
			}
			else
			{
				xb =  value - solver->cudaDofVector2[neighborEntities.template getEntityIndex< -1,  0 >()];
				xf = solver-> cudaDofVector2[neighborEntities.template getEntityIndex< 1,  0 >()] - value;
			}

			if( j == 0/* || (threadIdx.y == 0 && !(status & 3))*/ )
			{
				yb = value - solver->cudaDofVector2[neighborEntities.template getEntityIndex< 0,  1 >()] ;
				yf = solver->cudaDofVector2[neighborEntities.template getEntityIndex< 0,  1 >()] - value;
			}
			else if( j == solver->Mesh.getDimensions().y() - 1  /*|| (threadIdx.y == blockDim.y - 1 && !(status & 5)) */)
			{
				yb = value - solver->cudaDofVector2[neighborEntities.template getEntityIndex< 0,  -1 >()];
				yf = solver->cudaDofVector2[neighborEntities.template getEntityIndex< 0,  -1 >()] - value;
			}
			else
			{
				yb = value - solver->cudaDofVector2[neighborEntities.template getEntityIndex< 0, -1 >()];
				yf = solver-> cudaDofVector2[neighborEntities.template getEntityIndex< 0,  1 >()] - value;
			}
			__syncthreads();





			   if(sign(value) > 0.0)
			   {
				   xf = solver->negativePart(xf);

				   xb = solver->positivePart(xb);

				   yf = solver->negativePart(yf);

				   yb = solver->positivePart(yb);

			   }
			   else
			   {

				   xb = solver->negativePart(xb);

				   xf = solver->positivePart(xf);

				   yb = solver->negativePart(yb);

				   yf = solver->positivePart(yf);
			   }


			   if(xb > xf)
				   a = xb*solver->Mesh.template getSpaceStepsProducts< -1, 0 >();
			   else
				   a = xf*solver->Mesh.template getSpaceStepsProducts< -1, 0 >();

			   if(yb > yf)
				   b = yb*solver->Mesh.template getSpaceStepsProducts< 0, -1 >();
			   else
				   b = yf*solver->Mesh.template getSpaceStepsProducts< 0, -1 >();



//			grad = sqrt(0.5 * (xf*xf + xb*xb    +   yf*yf + yb*yb ) )*solver->Mesh.template getSpaceStepsProducts< -1, 0 >();

			grad = sqrt(/*0.5 **/ (a*a    +   b*b ) );

			fu = -1.0 * grad;

//			if((tau*fu+value)*value <=0 )
//			{
//				//			1 - with curve,  	2 - to the north of curve, 	4  - to the south of curve,
//				//								8 - to the east of curve, 	16 - to the west of curve.
//
//				if((threadIdx.x == 1 && !(status & 9)) && (blockIdx.x > 0) )
//					atomicMax(solver->reinitialize,1);
//				else if((threadIdx.x == blockDim.x - 2 && !(status & 17)) && (blockIdx.x < gridDim.x - 1) )
//					atomicMax(solver->reinitialize,1);
//				else if((threadIdx.y == 1 && !(status & 3)) && (blockIdx.y > 0) )
//					atomicMax(solver->reinitialize,1);
//				else if((threadIdx.y == blockDim.y - 2 && !(status & 5)) && (blockIdx.y < gridDim.y - 1) )
//					atomicMax(solver->reinitialize,1);
//			}

			solver->cudaDofVector2[Entity.getIndex()]  += tau*fu;
		}
	}
}




































template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
void tnlNarrowBand< tnlGrid< 2,MeshReal, Device, MeshIndex >, Real, Index > :: setupSquare1111( Index i, Index j)
{
	tnlGridEntity< tnlGrid< 2,double, TNL::Devices::Host, int >, 2, tnlGridEntityNoStencilStorage > Entity(Mesh);
	Entity.setCoordinates(CoordinatesType(i,j));
	Entity.refresh();
	tnlNeighborGridEntityGetter<tnlGridEntity< MeshType, 2, tnlGridEntityNoStencilStorage >,2> neighborEntities(Entity);
	cudaDofVector2[Entity.getIndex()]=fabsMin(INT_MAX,cudaDofVector2[Entity.getIndex()]);
	cudaDofVector2[neighborEntities.template getEntityIndex< 0,  1 >()]=fabsMin(INT_MAX,cudaDofVector2[neighborEntities.template getEntityIndex< 0,  1 >()]);
	cudaDofVector2[neighborEntities.template getEntityIndex< 1,  1 >()]=fabsMin(INT_MAX,cudaDofVector2[neighborEntities.template getEntityIndex< 1,  1 >()]);
	cudaDofVector2[neighborEntities.template getEntityIndex< 1,  0 >()]=fabsMin(INT_MAX,cudaDofVector2[neighborEntities.template getEntityIndex< 1,  0 >()]);

}


template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
void tnlNarrowBand< tnlGrid< 2,MeshReal, Device, MeshIndex >, Real, Index > :: setupSquare0000( Index i, Index j)
{
	tnlGridEntity< tnlGrid< 2,double, TNL::Devices::Host, int >, 2, tnlGridEntityNoStencilStorage > Entity(Mesh);
	Entity.setCoordinates(CoordinatesType(i,j));
	Entity.refresh();
	tnlNeighborGridEntityGetter<tnlGridEntity< MeshType, 2, tnlGridEntityNoStencilStorage >,2> neighborEntities(Entity);
	cudaDofVector2[Entity.getIndex()]=fabsMin(-INT_MAX,cudaDofVector2[Entity.getIndex()]);
	cudaDofVector2[neighborEntities.template getEntityIndex< 0,  1 >()]=fabsMin(-INT_MAX,cudaDofVector2[neighborEntities.template getEntityIndex< 0,  1 >()]);
	cudaDofVector2[neighborEntities.template getEntityIndex< 1,  1 >()]=fabsMin(-INT_MAX,cudaDofVector2[neighborEntities.template getEntityIndex< 1,  1 >()]);
	cudaDofVector2[neighborEntities.template getEntityIndex< 1,  0 >()]=fabsMin(-INT_MAX,cudaDofVector2[neighborEntities.template getEntityIndex< 1,  0 >()]);

}


template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
void tnlNarrowBand< tnlGrid< 2,MeshReal, Device, MeshIndex >, Real, Index > :: setupSquare1110( Index i, Index j)
{
	tnlGridEntity< tnlGrid< 2,double, TNL::Devices::Host, int >, 2, tnlGridEntityNoStencilStorage > Entity(Mesh);
	Entity.setCoordinates(CoordinatesType(i,j));
	Entity.refresh();
	tnlNeighborGridEntityGetter<tnlGridEntity< MeshType, 2, tnlGridEntityNoStencilStorage >,2> neighborEntities(Entity);
	Real al,be, a,b,c,s;
	al=abs(cudaDofVector[neighborEntities.template getEntityIndex< 1,  1 >()]/
			(cudaDofVector[neighborEntities.template getEntityIndex< 1,  0 >()]-
			 cudaDofVector[neighborEntities.template getEntityIndex< 1,  1 >()]));

	be=abs(cudaDofVector[neighborEntities.template getEntityIndex< 1,  1 >()]/
			(cudaDofVector[neighborEntities.template getEntityIndex< 0,  1 >()]-
			 cudaDofVector[neighborEntities.template getEntityIndex< 1,  1 >()]));

	a = be/al;
	b=1.0;
	c=-be;
	s= h/sqrt(a*a+b*b);


	cudaDofVector2[Entity.getIndex()]=fabsMin(abs(a*1+b*1+c)*s,cudaDofVector2[Entity.getIndex()]);
	cudaDofVector2[neighborEntities.template getEntityIndex< 0,  1 >()]=fabsMin(abs(a*0+b*1+c)*s,cudaDofVector2[neighborEntities.template getEntityIndex< 0,  1 >()]);
	cudaDofVector2[neighborEntities.template getEntityIndex< 1,  1 >()]=fabsMin(-abs(a*0+b*0+c)*s,cudaDofVector2[neighborEntities.template getEntityIndex< 1,  1 >()]);
	cudaDofVector2[neighborEntities.template getEntityIndex< 1,  0 >()]=fabsMin(abs(a*1+b*0+c)*s,cudaDofVector2[neighborEntities.template getEntityIndex< 1,  0 >()]);

}

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
void tnlNarrowBand< tnlGrid< 2,MeshReal, Device, MeshIndex >, Real, Index > :: setupSquare1101( Index i, Index j)
{
	tnlGridEntity< tnlGrid< 2,double, TNL::Devices::Host, int >, 2, tnlGridEntityNoStencilStorage > Entity(Mesh);
	Entity.setCoordinates(CoordinatesType(i,j));
	Entity.refresh();
	tnlNeighborGridEntityGetter<tnlGridEntity< MeshType, 2, tnlGridEntityNoStencilStorage >,2> neighborEntities(Entity);
	Real al,be, a,b,c,s;
	al=abs(cudaDofVector[neighborEntities.template getEntityIndex< 0,  1 >()]/
			(cudaDofVector[neighborEntities.template getEntityIndex< 1,  1 >()]-
			 cudaDofVector[neighborEntities.template getEntityIndex< 0,  1 >()]));

	be=abs(cudaDofVector[neighborEntities.template getEntityIndex< 0,  1 >()]/
			(cudaDofVector[Entity.getIndex()]-
			 cudaDofVector[neighborEntities.template getEntityIndex< 0,  1 >()]));

	a = be/al;
	b=1.0;
	c=-be;
	s= h/sqrt(a*a+b*b);


	cudaDofVector2[Entity.getIndex()]=fabsMin(abs(a*0+b*1+c)*s,cudaDofVector2[Entity.getIndex()]);
	cudaDofVector2[neighborEntities.template getEntityIndex< 0,  1 >()]=fabsMin(abs(a*0+b*0+c)*s,cudaDofVector2[neighborEntities.template getEntityIndex< 0,  1 >()]);
	cudaDofVector2[neighborEntities.template getEntityIndex< 1,  1 >()]=fabsMin(abs(a*1+b*0+c)*s,cudaDofVector2[neighborEntities.template getEntityIndex< 1,  1 >()]);
	cudaDofVector2[neighborEntities.template getEntityIndex< 1,  0 >()]=fabsMin(-abs(a*1+b*1+c)*s,cudaDofVector2[neighborEntities.template getEntityIndex< 1,  0 >()]);

}

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
void tnlNarrowBand< tnlGrid< 2,MeshReal, Device, MeshIndex >, Real, Index > :: setupSquare1011( Index i, Index j)
{
	tnlGridEntity< tnlGrid< 2,double, TNL::Devices::Host, int >, 2, tnlGridEntityNoStencilStorage > Entity(Mesh);
	Entity.setCoordinates(CoordinatesType(i,j));
	Entity.refresh();
	tnlNeighborGridEntityGetter<tnlGridEntity< MeshType, 2, tnlGridEntityNoStencilStorage >,2> neighborEntities(Entity);
	Real al,be, a,b,c,s;
	al=abs(cudaDofVector[neighborEntities.template getEntityIndex< 1,  0 >()]/
			(cudaDofVector[Entity.getIndex()]-
			 cudaDofVector[neighborEntities.template getEntityIndex< 1,  0 >()]));

	be=abs(cudaDofVector[neighborEntities.template getEntityIndex< 1,  0 >()]/
			(cudaDofVector[neighborEntities.template getEntityIndex< 1,  1 >()]-
			 cudaDofVector[neighborEntities.template getEntityIndex< 1,  0 >()]));

	a = be/al;
	b=1.0;
	c=-be;
	s= h/sqrt(a*a+b*b);


	cudaDofVector2[Entity.getIndex()]=fabsMin(abs(a*1+b*0+c)*s,cudaDofVector2[Entity.getIndex()]);
	cudaDofVector2[neighborEntities.template getEntityIndex< 0,  1 >()]=fabsMin(-abs(a*1+b*1+c)*s,cudaDofVector2[neighborEntities.template getEntityIndex< 0,  1 >()]);
	cudaDofVector2[neighborEntities.template getEntityIndex< 1,  1 >()]=fabsMin(abs(a*0+b*1+c)*s,cudaDofVector2[neighborEntities.template getEntityIndex< 1,  1 >()]);
	cudaDofVector2[neighborEntities.template getEntityIndex< 1,  0 >()]=fabsMin(abs(a*0+b*0+c)*s,cudaDofVector2[neighborEntities.template getEntityIndex< 1,  0 >()]);

}

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
void tnlNarrowBand< tnlGrid< 2,MeshReal, Device, MeshIndex >, Real, Index > :: setupSquare0111( Index i, Index j)
{
	tnlGridEntity< tnlGrid< 2,double, TNL::Devices::Host, int >, 2, tnlGridEntityNoStencilStorage > Entity(Mesh);
	Entity.setCoordinates(CoordinatesType(i,j));
	Entity.refresh();
	tnlNeighborGridEntityGetter<tnlGridEntity< MeshType, 2, tnlGridEntityNoStencilStorage >,2> neighborEntities(Entity);
	Real al,be, a,b,c,s;
	al=abs(cudaDofVector[Entity.getIndex()]/
			(cudaDofVector[neighborEntities.template getEntityIndex< 0,  1 >()]-
			 cudaDofVector[Entity.getIndex()]));

	be=abs(cudaDofVector[Entity.getIndex()]/
			(cudaDofVector[neighborEntities.template getEntityIndex< 1,  0 >()]-
			 cudaDofVector[Entity.getIndex()]));

	a = be/al;
	b=1.0;
	c=-be;
	s= h/sqrt(a*a+b*b);


	cudaDofVector2[Entity.getIndex()]=fabsMin(-abs(a*0+b*0+c)*s,cudaDofVector2[Entity.getIndex()]);
	cudaDofVector2[neighborEntities.template getEntityIndex< 0,  1 >()]=fabsMin(abs(a*1+b*0+c)*s,cudaDofVector2[neighborEntities.template getEntityIndex< 0,  1 >()]);
	cudaDofVector2[neighborEntities.template getEntityIndex< 1,  1 >()]=fabsMin(abs(a*1+b*1+c)*s,cudaDofVector2[neighborEntities.template getEntityIndex< 1,  1 >()]);
	cudaDofVector2[neighborEntities.template getEntityIndex< 1,  0 >()]=fabsMin(abs(a*0+b*1+c)*s,cudaDofVector2[neighborEntities.template getEntityIndex< 1,  0 >()]);

}


template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
void tnlNarrowBand< tnlGrid< 2,MeshReal, Device, MeshIndex >, Real, Index > :: setupSquare0001( Index i, Index j)
{
	tnlGridEntity< tnlGrid< 2,double, TNL::Devices::Host, int >, 2, tnlGridEntityNoStencilStorage > Entity(Mesh);
	Entity.setCoordinates(CoordinatesType(i,j));
	Entity.refresh();
	tnlNeighborGridEntityGetter<tnlGridEntity< MeshType, 2, tnlGridEntityNoStencilStorage >,2> neighborEntities(Entity);
	Real al,be, a,b,c,s;
	al=abs(cudaDofVector[neighborEntities.template getEntityIndex< 1,  1 >()]/
			(cudaDofVector[neighborEntities.template getEntityIndex< 1,  0 >()]-
			 cudaDofVector[neighborEntities.template getEntityIndex< 1,  1 >()]));

	be=abs(cudaDofVector[neighborEntities.template getEntityIndex< 1,  1 >()]/
			(cudaDofVector[neighborEntities.template getEntityIndex< 0,  1 >()]-
			 cudaDofVector[neighborEntities.template getEntityIndex< 1,  1 >()]));

	a = be/al;
	b=1.0;
	c=-be;
	s= h/sqrt(a*a+b*b);


	cudaDofVector2[Entity.getIndex()]=fabsMin(-abs(a*1+b*1+c)*s,cudaDofVector2[Entity.getIndex()]);
	cudaDofVector2[neighborEntities.template getEntityIndex< 0,  1 >()]=fabsMin(-abs(a*0+b*1+c)*s,cudaDofVector2[neighborEntities.template getEntityIndex< 0,  1 >()]);
	cudaDofVector2[neighborEntities.template getEntityIndex< 1,  1 >()]=fabsMin(abs(a*0+b*0+c)*s,cudaDofVector2[neighborEntities.template getEntityIndex< 1,  1 >()]);
	cudaDofVector2[neighborEntities.template getEntityIndex< 1,  0 >()]=fabsMin(-abs(a*1+b*0+c)*s,cudaDofVector2[neighborEntities.template getEntityIndex< 1,  0 >()]);

}

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
void tnlNarrowBand< tnlGrid< 2,MeshReal, Device, MeshIndex >, Real, Index > :: setupSquare0010( Index i, Index j)
{
	tnlGridEntity< tnlGrid< 2,double, TNL::Devices::Host, int >, 2, tnlGridEntityNoStencilStorage > Entity(Mesh);
	Entity.setCoordinates(CoordinatesType(i,j));
	Entity.refresh();
	tnlNeighborGridEntityGetter<tnlGridEntity< MeshType, 2, tnlGridEntityNoStencilStorage >,2> neighborEntities(Entity);
	Real al,be, a,b,c,s;
	al=abs(cudaDofVector[neighborEntities.template getEntityIndex< 0,  1 >()]/
			(cudaDofVector[neighborEntities.template getEntityIndex< 1,  1 >()]-
			 cudaDofVector[neighborEntities.template getEntityIndex< 0,  1 >()]));

	be=abs(cudaDofVector[neighborEntities.template getEntityIndex< 0,  1 >()]/
			(cudaDofVector[Entity.getIndex()]-
			 cudaDofVector[neighborEntities.template getEntityIndex< 0,  1 >()]));

	a = be/al;
	b=1.0;
	c=-be;
	s= h/sqrt(a*a+b*b);


	cudaDofVector2[Entity.getIndex()]=fabsMin(-abs(a*0+b*1+c)*s,cudaDofVector2[Entity.getIndex()]);
	cudaDofVector2[neighborEntities.template getEntityIndex< 0,  1 >()]=fabsMin(-abs(a*0+b*0+c)*s,cudaDofVector2[neighborEntities.template getEntityIndex< 0,  1 >()]);
	cudaDofVector2[neighborEntities.template getEntityIndex< 1,  1 >()]=fabsMin(-abs(a*1+b*0+c)*s,cudaDofVector2[neighborEntities.template getEntityIndex< 1,  1 >()]);
	cudaDofVector2[neighborEntities.template getEntityIndex< 1,  0 >()]=fabsMin(abs(a*1+b*1+c)*s,cudaDofVector2[neighborEntities.template getEntityIndex< 1,  0 >()]);

}

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
void tnlNarrowBand< tnlGrid< 2,MeshReal, Device, MeshIndex >, Real, Index > :: setupSquare0100( Index i, Index j)
{
	tnlGridEntity< tnlGrid< 2,double, TNL::Devices::Host, int >, 2, tnlGridEntityNoStencilStorage > Entity(Mesh);
	Entity.setCoordinates(CoordinatesType(i,j));
	Entity.refresh();
	tnlNeighborGridEntityGetter<tnlGridEntity< MeshType, 2, tnlGridEntityNoStencilStorage >,2> neighborEntities(Entity);
	Real al,be, a,b,c,s;
	al=abs(cudaDofVector[neighborEntities.template getEntityIndex< 1,  0 >()]/
			(cudaDofVector[Entity.getIndex()]-
			 cudaDofVector[neighborEntities.template getEntityIndex< 1,  0 >()]));

	be=abs(cudaDofVector[neighborEntities.template getEntityIndex< 1,  0 >()]/
			(cudaDofVector[neighborEntities.template getEntityIndex< 1,  1 >()]-
			 cudaDofVector[neighborEntities.template getEntityIndex< 1,  0 >()]));

	a = be/al;
	b=1.0;
	c=-be;
	s= h/sqrt(a*a+b*b);


	cudaDofVector2[Entity.getIndex()]=fabsMin(-abs(a*1+b*0+c)*s,cudaDofVector2[Entity.getIndex()]);
	cudaDofVector2[neighborEntities.template getEntityIndex< 0,  1 >()]=fabsMin(abs(a*1+b*1+c)*s,cudaDofVector2[neighborEntities.template getEntityIndex< 0,  1 >()]);
	cudaDofVector2[neighborEntities.template getEntityIndex< 1,  1 >()]=fabsMin(-abs(a*0+b*1+c)*s,cudaDofVector2[neighborEntities.template getEntityIndex< 1,  1 >()]);
	cudaDofVector2[neighborEntities.template getEntityIndex< 1,  0 >()]=fabsMin(-abs(a*0+b*0+c)*s,cudaDofVector2[neighborEntities.template getEntityIndex< 1,  0 >()]);

}

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
void tnlNarrowBand< tnlGrid< 2,MeshReal, Device, MeshIndex >, Real, Index > :: setupSquare1000( Index i, Index j)
{
	tnlGridEntity< tnlGrid< 2,double, TNL::Devices::Host, int >, 2, tnlGridEntityNoStencilStorage > Entity(Mesh);
	Entity.setCoordinates(CoordinatesType(i,j));
	Entity.refresh();
	tnlNeighborGridEntityGetter<tnlGridEntity< MeshType, 2, tnlGridEntityNoStencilStorage >,2> neighborEntities(Entity);
	Real al,be, a,b,c,s;
	al=abs(cudaDofVector[Entity.getIndex()]/
			(cudaDofVector[neighborEntities.template getEntityIndex< 0,  1 >()]-
			 cudaDofVector[Entity.getIndex()]));

	be=abs(cudaDofVector[Entity.getIndex()]/
			(cudaDofVector[neighborEntities.template getEntityIndex< 1,  0 >()]-
			 cudaDofVector[Entity.getIndex()]));

	a = be/al;
	b=1.0;
	c=-be;
	s= h/sqrt(a*a+b*b);


	cudaDofVector2[Entity.getIndex()]=fabsMin(abs(a*0+b*0+c)*s,cudaDofVector2[Entity.getIndex()]);
	cudaDofVector2[neighborEntities.template getEntityIndex< 0,  1 >()]=fabsMin(-abs(a*1+b*0+c)*s,cudaDofVector2[neighborEntities.template getEntityIndex< 0,  1 >()]);
	cudaDofVector2[neighborEntities.template getEntityIndex< 1,  1 >()]=fabsMin(-abs(a*1+b*1+c)*s,cudaDofVector2[neighborEntities.template getEntityIndex< 1,  1 >()]);
	cudaDofVector2[neighborEntities.template getEntityIndex< 1,  0 >()]=fabsMin(-abs(a*0+b*1+c)*s,cudaDofVector2[neighborEntities.template getEntityIndex< 1,  0 >()]);

}





template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
void tnlNarrowBand< tnlGrid< 2,MeshReal, Device, MeshIndex >, Real, Index > :: setupSquare1100( Index i, Index j)
{
	tnlGridEntity< tnlGrid< 2,double, TNL::Devices::Host, int >, 2, tnlGridEntityNoStencilStorage > Entity(Mesh);
	Entity.setCoordinates(CoordinatesType(i,j));
	Entity.refresh();
	tnlNeighborGridEntityGetter<tnlGridEntity< MeshType, 2, tnlGridEntityNoStencilStorage >,2> neighborEntities(Entity);
	Real al,be, a,b,c,s;
	al=abs(cudaDofVector[Entity.getIndex()]/
			(cudaDofVector[neighborEntities.template getEntityIndex< 0,  1 >()]-
			 cudaDofVector[Entity.getIndex()]));

	be=abs(cudaDofVector[neighborEntities.template getEntityIndex< 1,  0 >()]/
			(cudaDofVector[neighborEntities.template getEntityIndex< 1,  1 >()]-
			 cudaDofVector[neighborEntities.template getEntityIndex< 1,  0 >()]));

	a = al-be;
	b=1.0;
	c=-al;
	s= h/sqrt(a*a+b*b);


	cudaDofVector2[Entity.getIndex()]=fabsMin(abs(a*0+b*0+c)*s,cudaDofVector2[Entity.getIndex()]);
	cudaDofVector2[neighborEntities.template getEntityIndex< 0,  1 >()]=fabsMin(-abs(a*0+b*1+c)*s,cudaDofVector2[neighborEntities.template getEntityIndex< 0,  1 >()]);
	cudaDofVector2[neighborEntities.template getEntityIndex< 1,  1 >()]=fabsMin(-abs(a*1+b*1+c)*s,cudaDofVector2[neighborEntities.template getEntityIndex< 1,  1 >()]);
	cudaDofVector2[neighborEntities.template getEntityIndex< 1,  0 >()]=fabsMin(abs(a*1+b*0+c)*s,cudaDofVector2[neighborEntities.template getEntityIndex< 1,  0 >()]);

}

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
void tnlNarrowBand< tnlGrid< 2,MeshReal, Device, MeshIndex >, Real, Index > :: setupSquare1010( Index i, Index j)
{
	tnlGridEntity< tnlGrid< 2,double, TNL::Devices::Host, int >, 2, tnlGridEntityNoStencilStorage > Entity(Mesh);
	Entity.setCoordinates(CoordinatesType(i,j));
	Entity.refresh();
	tnlNeighborGridEntityGetter<tnlGridEntity< MeshType, 2, tnlGridEntityNoStencilStorage >,2> neighborEntities(Entity);
	Real al,be, a,b,c,s;
	al=abs(cudaDofVector[Entity.getIndex()]/
			(cudaDofVector[neighborEntities.template getEntityIndex< 1,  0 >()]-
			 cudaDofVector[Entity.getIndex()]));

	be=abs(cudaDofVector[neighborEntities.template getEntityIndex< 0,  1 >()]/
			(cudaDofVector[neighborEntities.template getEntityIndex< 1,  1 >()]-
			 cudaDofVector[neighborEntities.template getEntityIndex< 0,  1 >()]));

	a = al-be;
	b=1.0;
	c=-be;
	s= h/sqrt(a*a+b*b);


	cudaDofVector2[Entity.getIndex()]=fabsMin(abs(a*0+b*0+c)*s,cudaDofVector2[Entity.getIndex()]);
	cudaDofVector2[neighborEntities.template getEntityIndex< 0,  1 >()]=fabsMin(abs(a*1+b*0+c)*s,cudaDofVector2[neighborEntities.template getEntityIndex< 0,  1 >()]);
	cudaDofVector2[neighborEntities.template getEntityIndex< 1,  1 >()]=fabsMin(-abs(a*1+b*1+c)*s,cudaDofVector2[neighborEntities.template getEntityIndex< 1,  1 >()]);
	cudaDofVector2[neighborEntities.template getEntityIndex< 1,  0 >()]=fabsMin(-abs(a*0+b*1+c)*s,cudaDofVector2[neighborEntities.template getEntityIndex< 1,  0 >()]);

}

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
void tnlNarrowBand< tnlGrid< 2,MeshReal, Device, MeshIndex >, Real, Index > :: setupSquare1001( Index i, Index j)
{
	tnlGridEntity< tnlGrid< 2,double, TNL::Devices::Host, int >, 2, tnlGridEntityNoStencilStorage > Entity(Mesh);
	Entity.setCoordinates(CoordinatesType(i,j));
	Entity.refresh();
	tnlNeighborGridEntityGetter<tnlGridEntity< MeshType, 2, tnlGridEntityNoStencilStorage >,2> neighborEntities(Entity);
	cudaDofVector2[Entity.getIndex()]=fabsMin(cudaDofVector[Entity.getIndex()],cudaDofVector2[Entity.getIndex()]);
	cudaDofVector2[neighborEntities.template getEntityIndex< 0,  1 >()]=fabsMin(cudaDofVector[neighborEntities.template getEntityIndex< 0,  1 >()],cudaDofVector2[neighborEntities.template getEntityIndex< 0,  1 >()]);
	cudaDofVector2[neighborEntities.template getEntityIndex< 1,  1 >()]=fabsMin(cudaDofVector[neighborEntities.template getEntityIndex< 1,  1 >()],cudaDofVector2[neighborEntities.template getEntityIndex< 1,  1 >()]);
	cudaDofVector2[neighborEntities.template getEntityIndex< 1,  0 >()]=fabsMin(cudaDofVector[neighborEntities.template getEntityIndex< 1,  0 >()],cudaDofVector2[neighborEntities.template getEntityIndex< 1,  0 >()]);

}







template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
void tnlNarrowBand< tnlGrid< 2,MeshReal, Device, MeshIndex >, Real, Index > :: setupSquare0011( Index i, Index j)
{
	tnlGridEntity< tnlGrid< 2,double, TNL::Devices::Host, int >, 2, tnlGridEntityNoStencilStorage > Entity(Mesh);
	Entity.setCoordinates(CoordinatesType(i,j));
	Entity.refresh();
	tnlNeighborGridEntityGetter<tnlGridEntity< MeshType, 2, tnlGridEntityNoStencilStorage >,2> neighborEntities(Entity);
	Real al,be, a,b,c,s;
	al=abs(cudaDofVector[Entity.getIndex()]/
			(cudaDofVector[neighborEntities.template getEntityIndex< 0,  1 >()]-
			 cudaDofVector[Entity.getIndex()]));

	be=abs(cudaDofVector[neighborEntities.template getEntityIndex< 1,  0 >()]/
			(cudaDofVector[neighborEntities.template getEntityIndex< 1,  1 >()]-
			 cudaDofVector[neighborEntities.template getEntityIndex< 1,  0 >()]));

	a = al-be;
	b=1.0;
	c=-al;
	s= h/sqrt(a*a+b*b);


	cudaDofVector2[Entity.getIndex()]=fabsMin(-abs(a*0+b*0+c)*s,cudaDofVector2[Entity.getIndex()]);
	cudaDofVector2[neighborEntities.template getEntityIndex< 0,  1 >()]=fabsMin(abs(a*0+b*1+c)*s,cudaDofVector2[neighborEntities.template getEntityIndex< 0,  1 >()]);
	cudaDofVector2[neighborEntities.template getEntityIndex< 1,  1 >()]=fabsMin(abs(a*1+b*1+c)*s,cudaDofVector2[neighborEntities.template getEntityIndex< 1,  1 >()]);
	cudaDofVector2[neighborEntities.template getEntityIndex< 1,  0 >()]=fabsMin(-abs(a*1+b*0+c)*s,cudaDofVector2[neighborEntities.template getEntityIndex< 1,  0 >()]);

}

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
void tnlNarrowBand< tnlGrid< 2,MeshReal, Device, MeshIndex >, Real, Index > :: setupSquare0101( Index i, Index j)
{
	tnlGridEntity< tnlGrid< 2,double, TNL::Devices::Host, int >, 2, tnlGridEntityNoStencilStorage > Entity(Mesh);
	Entity.setCoordinates(CoordinatesType(i,j));
	Entity.refresh();
	tnlNeighborGridEntityGetter<tnlGridEntity< MeshType, 2, tnlGridEntityNoStencilStorage >,2> neighborEntities(Entity);
	Real al,be, a,b,c,s;
	al=abs(cudaDofVector[Entity.getIndex()]/
			(cudaDofVector[neighborEntities.template getEntityIndex< 1,  0 >()]-
			 cudaDofVector[Entity.getIndex()]));

	be=abs(cudaDofVector[neighborEntities.template getEntityIndex< 0,  1 >()]/
			(cudaDofVector[neighborEntities.template getEntityIndex< 1,  1 >()]-
			 cudaDofVector[neighborEntities.template getEntityIndex< 0,  1 >()]));

	a = al-be;
	b=1.0;
	c=-be;
	s= h/sqrt(a*a+b*b);


	cudaDofVector2[Entity.getIndex()]=fabsMin(-abs(a*0+b*0+c)*s,cudaDofVector2[Entity.getIndex()]);
	cudaDofVector2[neighborEntities.template getEntityIndex< 0,  1 >()]=fabsMin(-abs(a*1+b*0+c)*s,cudaDofVector2[neighborEntities.template getEntityIndex< 0,  1 >()]);
	cudaDofVector2[neighborEntities.template getEntityIndex< 1,  1 >()]=fabsMin(abs(a*1+b*1+c)*s,cudaDofVector2[neighborEntities.template getEntityIndex< 1,  1 >()]);
	cudaDofVector2[neighborEntities.template getEntityIndex< 1,  0 >()]=fabsMin(abs(a*0+b*1+c)*s,cudaDofVector2[neighborEntities.template getEntityIndex< 1,  0 >()]);

}

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
void tnlNarrowBand< tnlGrid< 2,MeshReal, Device, MeshIndex >, Real, Index > :: setupSquare0110( Index i, Index j)
{
	tnlGridEntity< tnlGrid< 2,double, TNL::Devices::Host, int >, 2, tnlGridEntityNoStencilStorage > Entity(Mesh);
	Entity.setCoordinates(CoordinatesType(i,j));
	Entity.refresh();
	tnlNeighborGridEntityGetter<tnlGridEntity< MeshType, 2, tnlGridEntityNoStencilStorage >,2> neighborEntities(Entity);
	cudaDofVector2[Entity.getIndex()]=fabsMin(cudaDofVector[Entity.getIndex()],cudaDofVector2[Entity.getIndex()]);
	cudaDofVector2[neighborEntities.template getEntityIndex< 0,  1 >()]=fabsMin(cudaDofVector[neighborEntities.template getEntityIndex< 0,  1 >()],cudaDofVector2[neighborEntities.template getEntityIndex< 0,  1 >()]);
	cudaDofVector2[neighborEntities.template getEntityIndex< 1,  1 >()]=fabsMin(cudaDofVector[neighborEntities.template getEntityIndex< 1,  1 >()],cudaDofVector2[neighborEntities.template getEntityIndex< 1,  1 >()]);
	cudaDofVector2[neighborEntities.template getEntityIndex< 1,  0 >()]=fabsMin(cudaDofVector[neighborEntities.template getEntityIndex< 1,  0 >()],cudaDofVector2[neighborEntities.template getEntityIndex< 1,  0 >()]);
}
#endif




#endif /* TNLNARROWBAND_IMPL_H_ */
