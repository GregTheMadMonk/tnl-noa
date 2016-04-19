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

#define NARROWBAND_SUBGRID_SIZE 8

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
tnlString tnlNarrowBand< tnlGrid< 2,MeshReal, Device, MeshIndex >, Real, Index > :: getType()
{
	   return tnlString( "tnlNarrowBand< " ) +
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
bool tnlNarrowBand< tnlGrid< 2,MeshReal, Device, MeshIndex >, Real, Index > :: init( const tnlParameterContainer& parameters )
{
	const tnlString& meshFile = parameters.getParameter< tnlString >( "mesh" );

	if( ! Mesh.load( meshFile ) )
	{
		   cerr << "I am not able to load the mesh from the file " << meshFile << "." << endl;
		   return false;
	}


	const tnlString& initialCondition = parameters.getParameter <tnlString>("initial-condition");
	if( ! dofVector.load( initialCondition ) )
	{
		   cerr << "I am not able to load the initial condition from the file " << meshFile << "." << endl;
		   return false;
	}

	h = Mesh.template getSpaceStepsProducts< 1, 0 >();
	//Entity.refresh();
	counter = 0;

	const tnlString& exact_input = parameters.getParameter< tnlString >( "exact-input" );

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

	cudaMalloc(&(cudaStatusVector),  statusGridSize*statusGridSize* sizeof(int));
//	cudaMemcpy(cudaDofVector, this->dofVector.getData().getData(),  statusGridSize*statusGridSize* sizeof(int)), cudaMemcpyHostToDevice);

	cudaMalloc(&reinitialize, sizeof(int));


	cudaMalloc(&(this->cudaSolver), sizeof(tnlNarrowBand< tnlGrid< 2,MeshReal, Device, MeshIndex >, Real, Index >));
	cudaMemcpy(this->cudaSolver, this,sizeof(tnlNarrowBand< tnlGrid< 2,MeshReal, Device, MeshIndex >, Real, Index >), cudaMemcpyHostToDevice);

#endif

	int n = Mesh.getDimensions().x();

	dim3 threadsPerBlock2(NARROWBAND_SUBGRID_SIZE, NARROWBAND_SUBGRID_SIZE);
	dim3 numBlocks2(((Mesh.getDimensions().x() + NARROWBAND_SUBGRID_SIZE-1 ) / NARROWBAND_SUBGRID_SIZE) ,((Mesh.getDimensions().x() + NARROWBAND_SUBGRID_SIZE-1 ) / NARROWBAND_SUBGRID_SIZE));
	initSetupGridCUDA<<<numBlocks2,threadsPerBlock2>>>(this->cudaSolver);
	cudaDeviceSynchronize();
	checkCudaDevice;
	initSetupGrid2CUDA<<<numBlocks2,1>>>(this->cudaSolver);
	cudaDeviceSynchronize();
	checkCudaDevice;


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
bool tnlNarrowBand< tnlGrid< 2,MeshReal, Device, MeshIndex >, Real, Index > :: run()
{

	int n = Mesh.getDimensions().x();
	dim3 threadsPerBlockFS(1, 512);
	dim3 numBlocksFS(4,1);
	dim3 threadsPerBlockNB(NARROWBAND_SUBGRID_SIZE, NARROWBAND_SUBGRID_SIZE);
	dim3 numBlocksNB(n/NARROWBAND_SUBGRID_SIZE + 1,n/NARROWBAND_SUBGRID_SIZE + 1);

	double time = 0.0;
	int reinit = 0;

	runCUDA<<<numBlocksFS,threadsPerBlockFS>>>(this->cudaSolver,0,0);
	cudaDeviceSynchronize();
	checkCudaDevice;
	while(time < finalTime)
	{
		if(tau+time > finalTime)
			tau=finalTime-time;

		runNarrowBandCUDA<<<numBlocksNB,threadsPerBlockNB>>>(this->cudaSolver,tau);
		cudaDeviceSynchronize();
		checkCudaDevice;

		time += tau;


		cudaMemcpy(&reinit, this->reinitialize, sizeof(int), cudaMemcpyDeviceToHost);
		if(reinit != 0)
		{
			initSetupGridCUDA<<<numBlocksNB,threadsPerBlockNB>>>(this->cudaSolver);
			cudaDeviceSynchronize();
			checkCudaDevice;
			initSetupGridCUDA<<<numBlocksNB,1>>>(this->cudaSolver);
			cudaDeviceSynchronize();
			checkCudaDevice;
			runCUDA<<<numBlocksFS,threadsPerBlockFS>>>(this->cudaSolver,0,0);
			cudaDeviceSynchronize();
			checkCudaDevice;
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
	if(cudaStatusVector[i/NARROWBAND_SUBGRID_SIZE + (j/NARROWBAND_SUBGRID_SIZE) * ((Mesh.getDimensions().x() + NARROWBAND_SUBGRID_SIZE-1 ) / NARROWBAND_SUBGRID_SIZE)] != 0)
	{
		tnlGridEntity< tnlGrid< 2,double, tnlHost, int >, 2, tnlGridEntityNoStencilStorage > Entity(Mesh);
		Entity.setCoordinates(CoordinatesType(i,j));
		Entity.refresh();
		tnlNeighbourGridEntityGetter<tnlGridEntity< MeshType, 2, tnlGridEntityNoStencilStorage >,2> neighbourEntities(Entity);
		Real value = cudaDofVector2[Entity.getIndex()];
		Real a,b, tmp;

		if( i == 0 )
			a = cudaDofVector2[neighbourEntities.template getEntityIndex< 1,  0 >()];
		else if( i == Mesh.getDimensions().x() - 1 )
			a = cudaDofVector2[neighbourEntities.template getEntityIndex< -1,  0 >()];
		else
		{
			a = fabsMin( cudaDofVector2[neighbourEntities.template getEntityIndex< -1,  0 >()],
					 cudaDofVector2[neighbourEntities.template getEntityIndex< 1,  0 >()] );
		}

		if( j == 0 )
			b = cudaDofVector2[neighbourEntities.template getEntityIndex< 0,  1 >()];
		else if( j == Mesh.getDimensions().y() - 1 )
			b = cudaDofVector2[neighbourEntities.template getEntityIndex< 0,  -1 >()];
		else
		{
			b = fabsMin( cudaDofVector2[neighbourEntities.template getEntityIndex< 0,  -1 >()],
					 cudaDofVector2[neighbourEntities.template getEntityIndex< 0,  1 >()] );
		}


		if(abs(a-b) >= h)
			tmp = fabsMin(a,b) + Sign(value)*h;
		else
			tmp = 0.5 * (a + b + Sign(value)*sqrt(2.0 * h * h - (a - b) * (a - b) ) );

	//	cudaDofVector2[Entity.getIndex()]  = fabsMin(value, tmp);
		atomicFabsMin(&(cudaDofVector2[Entity.getIndex()]), tmp);
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

	tnlGridEntity< tnlGrid< 2,double, tnlHost, int >, 2, tnlGridEntityNoStencilStorage > Entity(Mesh);

	Entity.setCoordinates(CoordinatesType(i,j));
	Entity.refresh();
	tnlNeighbourGridEntityGetter<tnlGridEntity< MeshType, 2, tnlGridEntityNoStencilStorage >,2> neighbourEntities(Entity);

	int gid = Entity.getIndex();

	cudaDofVector2[gid] = INT_MAX*Sign(cudaDofVector[gid]);
//
//	if(abs(cudaDofVector[gid]) < 1.01*h)
//		cudaDofVector2[gid] = cudaDofVector[gid];





	if(i+1 < Mesh.getDimensions().x() && j+1 < Mesh.getDimensions().y() )
	{
		if(cudaDofVector[Entity.getIndex()] > 0)
		{
			if(cudaDofVector[neighbourEntities.template getEntityIndex< 1,  0 >()] > 0)
			{
				if(cudaDofVector[neighbourEntities.template getEntityIndex< 0,  1 >()] > 0)
				{
					if(cudaDofVector[neighbourEntities.template getEntityIndex< 1,  1 >()] > 0)
						setupSquare1111(i,j);
					else
						setupSquare1110(i,j);
				}
				else
				{
					if(cudaDofVector[neighbourEntities.template getEntityIndex< 1,  1 >()] > 0)
						setupSquare1101(i,j);
					else
						setupSquare1100(i,j);
				}
			}
			else
			{
				if(cudaDofVector[neighbourEntities.template getEntityIndex< 0,  1 >()] > 0)
				{
					if(cudaDofVector[neighbourEntities.template getEntityIndex< 1,  1 >()] > 0)
						setupSquare1011(i,j);
					else
						setupSquare1010(i,j);
				}
				else
				{
					if(cudaDofVector[neighbourEntities.template getEntityIndex< 1,  1 >()] > 0)
						setupSquare1001(i,j);
					else
						setupSquare1000(i,j);
				}
			}
		}
		else
		{
			if(cudaDofVector[neighbourEntities.template getEntityIndex< 1,  0 >()] > 0)
			{
				if(cudaDofVector[neighbourEntities.template getEntityIndex< 0,  1 >()] > 0)
				{
					if(cudaDofVector[neighbourEntities.template getEntityIndex< 1,  1 >()] > 0)
						setupSquare0111(i,j);
					else
						setupSquare0110(i,j);
				}
				else
				{
					if(cudaDofVector[neighbourEntities.template getEntityIndex< 1,  1 >()] > 0)
						setupSquare0101(i,j);
					else
						setupSquare0100(i,j);
				}
			}
			else
			{
				if(cudaDofVector[neighbourEntities.template getEntityIndex< 0,  1 >()] > 0)
				{
					if(cudaDofVector[neighbourEntities.template getEntityIndex< 1,  1 >()] > 0)
						setupSquare0011(i,j);
					else
						setupSquare0010(i,j);
				}
				else
				{
					if(cudaDofVector[neighbourEntities.template getEntityIndex< 1,  1 >()] > 0)
						setupSquare0001(i,j);
					else
						setupSquare0000(i,j);
				}
			}
		}

	}

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



__global__ void runCUDA(tnlNarrowBand< tnlGrid< 2,double, tnlHost, int >, double, int >* solver, int sweep, int i)
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


__global__ void initCUDA(tnlNarrowBand< tnlGrid< 2,double, tnlHost, int >, double, int >* solver)
{


	int gx = threadIdx.x + blockDim.x*blockIdx.x;
	int gy = blockDim.y*blockIdx.y + threadIdx.y;


	if(solver->Mesh.getDimensions().x() > gx && solver->Mesh.getDimensions().y() > gy)
	{
		solver->initGrid();
	}


}





__global__ void initSetupGridCUDA(tnlNarrowBand< tnlGrid< 2,double, tnlHost, int >, double, int >* solver)
{
	__shared__ double u0;
	if(threadIdx.x+threadIdx.y == 0)
	{
		if(blockIdx.x+blockIdx.y == 0)
			*(solver->reinitialize) = 0;

		solver->cudaStatusVector[blockIdx.x + gridDim.x*blockIdx.y] = 0;

		u0 = solver->cudaDofVector2[(blockDim.y*blockIdx.y + 0)*blockDim.x*gridDim.x + blockDim.x*blockIdx.x + 0];
	}
	__syncthreads();

	double u = solver->cudaDofVector2[(blockDim.y*blockIdx.y + threadIdx.y)*blockDim.x*gridDim.x + blockDim.x*blockIdx.x + threadIdx.x];

	if(u*u0 <=0.0)
		atomicMax(&(solver->cudaStatusVector[blockIdx.x + gridDim.x*blockIdx.y]),1);
}



// run this with one thread per block
__global__ void initSetupGrid2CUDA(tnlNarrowBand< tnlGrid< 2,double, tnlHost, int >, double, int >* solver)
{
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





__global__ void runNarrowBandCUDA(tnlNarrowBand< tnlGrid< 2,double, tnlHost, int >, double, int >* solver, double tau)
{
	int gid = (blockDim.y*blockIdx.y + threadIdx.y)*blockDim.x*gridDim.x + blockDim.x*blockIdx.x + threadIdx.x;
	int i = threadIdx.x + blockIdx.x*blockDim.x;
	int j = threadIdx.y + blockIdx.y*blockDim.y;

	int blockID = blockIdx.x + blockIdx.y*gridDim.x;


	if(solver->cudaStatusVector[blockID/*i/NARROWBAND_SUBGRID_SIZE + (j/NARROWBAND_SUBGRID_SIZE) * ((Mesh.getDimensions().x() + NARROWBAND_SUBGRID_SIZE-1 ) / NARROWBAND_SUBGRID_SIZE)*/] != 0)
	{
		tnlGridEntity< tnlGrid< 2,double, tnlHost, int >, 2, tnlGridEntityNoStencilStorage > Entity(solver->Mesh);
		Entity.setCoordinates(tnlStaticVector<2,double>(i,j));
		Entity.refresh();
		tnlNeighbourGridEntityGetter<tnlGridEntity< tnlGrid< 2,double, tnlHost, int >, 2, tnlGridEntityNoStencilStorage >,2> neighbourEntities(Entity);
		double value = solver->cudaDofVector2[Entity.getIndex()];
		double xf,xb,yf,yb, tmp, fu;

		if( i == 0 )
			yb = yf = solver->cudaDofVector2[neighbourEntities.template getEntityIndex< 1,  0 >()];
		else if( i == solver->Mesh.getDimensions().x() - 1 )
			yb = yf = solver->cudaDofVector2[neighbourEntities.template getEntityIndex< -1,  0 >()];
		else
		{
			yb =  solver->cudaDofVector2[neighbourEntities.template getEntityIndex< -1,  0 >()];
			yf = solver-> cudaDofVector2[neighbourEntities.template getEntityIndex< 1,  0 >()];
		}

		if( j == 0 )
			xb = xf = solver->cudaDofVector2[neighbourEntities.template getEntityIndex< 0,  1 >()];
		else if( j == solver->Mesh.getDimensions().y() - 1 )
			xb = xf = solver->cudaDofVector2[neighbourEntities.template getEntityIndex< 0, -1 >()];
		else
		{
			xb =  solver->cudaDofVector2[neighbourEntities.template getEntityIndex< 0, -1 >()];
			xf = solver-> cudaDofVector2[neighbourEntities.template getEntityIndex< 0,  1 >()];
		}


		tmp = sqrt(0.5 * (xf*xf + xb*xb    +   yf*yf + yb*yb ) );

		fu = 1.0 * tmp;

		if((tau*fu+value)*value <=0 )
		{
			int status = solver->cudaStatusVector[blockID];
//			1 - with curve,  	2 - to the north of curve, 	4  - to the south of curve,
//								8 - to the east of curve, 	16 - to the west of curve.

			if(threadIdx.x == 0 && !(status & 8) && (blockIdx.x > 0) )
				atomicMax(solver->reinitialize,1);
			else if(threadIdx.x == blockDim.x-1 && !(status & 16) && (blockIdx.x < gridDim.x - 1) )
				atomicMax(solver->reinitialize,1);
			else if(threadIdx.y == 0 && !(status & 2) && (blockIdx.y > 0) )
				atomicMax(solver->reinitialize,1);
			else if(threadIdx.y == blockDim.y-1 && !(status & 4) && (blockIdx.y < gridDim.y - 1) )
				atomicMax(solver->reinitialize,1);

//			if(blockIdx.x < gridDim.x - 1)
//			{
//				atomicMax( &cudaStatusVector[(i+2)/NARROWBAND_SUBGRID_SIZE + (j/NARROWBAND_SUBGRID_SIZE) * ((Mesh.getDimensions().x() + NARROWBAND_SUBGRID_SIZE-1 ) / NARROWBAND_SUBGRID_SIZE)]  ,1);
//				if(blockIdx.y < gridDim.y -1)
//					atomicMax( &cudaStatusVector[(i+2)/NARROWBAND_SUBGRID_SIZE + (j+2/NARROWBAND_SUBGRID_SIZE) * ((Mesh.getDimensions().x() + NARROWBAND_SUBGRID_SIZE-1 ) / NARROWBAND_SUBGRID_SIZE)]  ,1);
//				if(blockIdx.y > 0)
//					atomicMax( &cudaStatusVector[(i+2)/NARROWBAND_SUBGRID_SIZE + (j-2/NARROWBAND_SUBGRID_SIZE) * ((Mesh.getDimensions().x() + NARROWBAND_SUBGRID_SIZE-1 ) / NARROWBAND_SUBGRID_SIZE)]  ,1);
//
//			}
//			if(blockIdx.x > 0)
//			{
//				atomicMax( &cudaStatusVector[(i-2)/NARROWBAND_SUBGRID_SIZE + (j/NARROWBAND_SUBGRID_SIZE) * ((Mesh.getDimensions().x() + NARROWBAND_SUBGRID_SIZE-1 ) / NARROWBAND_SUBGRID_SIZE)]  ,1);
//				if(blockIdx.y < gridDim.y -1)
//					atomicMax( &cudaStatusVector[(i-2)/NARROWBAND_SUBGRID_SIZE + (j+2/NARROWBAND_SUBGRID_SIZE) * ((Mesh.getDimensions().x() + NARROWBAND_SUBGRID_SIZE-1 ) / NARROWBAND_SUBGRID_SIZE)]  ,1);
//				if(blockIdx.y > 0)
//					atomicMax( &cudaStatusVector[(i-2)/NARROWBAND_SUBGRID_SIZE + (j-2/NARROWBAND_SUBGRID_SIZE) * ((Mesh.getDimensions().x() + NARROWBAND_SUBGRID_SIZE-1 ) / NARROWBAND_SUBGRID_SIZE)]  ,1);
//
//			}
//			if(blockIdx.y < gridDim.y -1)
//				atomicMax( &cudaStatusVector[i/NARROWBAND_SUBGRID_SIZE + ((j+2)/NARROWBAND_SUBGRID_SIZE) * ((Mesh.getDimensions().x() + NARROWBAND_SUBGRID_SIZE-1 ) / NARROWBAND_SUBGRID_SIZE)]  ,1);
//			if(blockIdx.y > 0)
//				atomicMax( &cudaStatusVector[i/NARROWBAND_SUBGRID_SIZE + ((j-2)/NARROWBAND_SUBGRID_SIZE) * ((Mesh.getDimensions().x() + NARROWBAND_SUBGRID_SIZE-1 ) / NARROWBAND_SUBGRID_SIZE)]  ,1);
		}


		solver->cudaDofVector2[Entity.getIndex()]  = value+tau*fu;
	}
}




































template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
void tnlNarrowBand< tnlGrid< 2,MeshReal, Device, MeshIndex >, Real, Index > :: setupSquare1111( Index i, Index j)
{
	tnlGridEntity< tnlGrid< 2,double, tnlHost, int >, 2, tnlGridEntityNoStencilStorage > Entity(Mesh);
	Entity.setCoordinates(CoordinatesType(i,j));
	Entity.refresh();
	tnlNeighbourGridEntityGetter<tnlGridEntity< MeshType, 2, tnlGridEntityNoStencilStorage >,2> neighbourEntities(Entity);
	cudaDofVector2[Entity.getIndex()]=fabsMin(INT_MAX,cudaDofVector2[Entity.getIndex()]);
	cudaDofVector2[neighbourEntities.template getEntityIndex< 0,  1 >()]=fabsMin(INT_MAX,cudaDofVector2[neighbourEntities.template getEntityIndex< 0,  1 >()]);
	cudaDofVector2[neighbourEntities.template getEntityIndex< 1,  1 >()]=fabsMin(INT_MAX,cudaDofVector2[neighbourEntities.template getEntityIndex< 1,  1 >()]);
	cudaDofVector2[neighbourEntities.template getEntityIndex< 1,  0 >()]=fabsMin(INT_MAX,cudaDofVector2[neighbourEntities.template getEntityIndex< 1,  0 >()]);

}


template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
void tnlNarrowBand< tnlGrid< 2,MeshReal, Device, MeshIndex >, Real, Index > :: setupSquare0000( Index i, Index j)
{
	tnlGridEntity< tnlGrid< 2,double, tnlHost, int >, 2, tnlGridEntityNoStencilStorage > Entity(Mesh);
	Entity.setCoordinates(CoordinatesType(i,j));
	Entity.refresh();
	tnlNeighbourGridEntityGetter<tnlGridEntity< MeshType, 2, tnlGridEntityNoStencilStorage >,2> neighbourEntities(Entity);
	cudaDofVector2[Entity.getIndex()]=fabsMin(-INT_MAX,cudaDofVector2[Entity.getIndex()]);
	cudaDofVector2[neighbourEntities.template getEntityIndex< 0,  1 >()]=fabsMin(-INT_MAX,cudaDofVector2[neighbourEntities.template getEntityIndex< 0,  1 >()]);
	cudaDofVector2[neighbourEntities.template getEntityIndex< 1,  1 >()]=fabsMin(-INT_MAX,cudaDofVector2[neighbourEntities.template getEntityIndex< 1,  1 >()]);
	cudaDofVector2[neighbourEntities.template getEntityIndex< 1,  0 >()]=fabsMin(-INT_MAX,cudaDofVector2[neighbourEntities.template getEntityIndex< 1,  0 >()]);

}


template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
void tnlNarrowBand< tnlGrid< 2,MeshReal, Device, MeshIndex >, Real, Index > :: setupSquare1110( Index i, Index j)
{
	tnlGridEntity< tnlGrid< 2,double, tnlHost, int >, 2, tnlGridEntityNoStencilStorage > Entity(Mesh);
	Entity.setCoordinates(CoordinatesType(i,j));
	Entity.refresh();
	tnlNeighbourGridEntityGetter<tnlGridEntity< MeshType, 2, tnlGridEntityNoStencilStorage >,2> neighbourEntities(Entity);
	Real al,be, a,b,c,s;
	al=abs(cudaDofVector[neighbourEntities.template getEntityIndex< 1,  1 >()]/
			(cudaDofVector[neighbourEntities.template getEntityIndex< 1,  0 >()]-
			 cudaDofVector[neighbourEntities.template getEntityIndex< 1,  1 >()]));

	be=abs(cudaDofVector[neighbourEntities.template getEntityIndex< 1,  1 >()]/
			(cudaDofVector[neighbourEntities.template getEntityIndex< 0,  1 >()]-
			 cudaDofVector[neighbourEntities.template getEntityIndex< 1,  1 >()]));

	a = be/al;
	b=1.0;
	c=-be;
	s= h/sqrt(a*a+b*b);


	cudaDofVector2[Entity.getIndex()]=fabsMin(abs(a*1+b*1+c)*s,cudaDofVector2[Entity.getIndex()]);
	cudaDofVector2[neighbourEntities.template getEntityIndex< 0,  1 >()]=fabsMin(abs(a*0+b*1+c)*s,cudaDofVector2[neighbourEntities.template getEntityIndex< 0,  1 >()]);
	cudaDofVector2[neighbourEntities.template getEntityIndex< 1,  1 >()]=fabsMin(-abs(a*0+b*0+c)*s,cudaDofVector2[neighbourEntities.template getEntityIndex< 1,  1 >()]);
	cudaDofVector2[neighbourEntities.template getEntityIndex< 1,  0 >()]=fabsMin(abs(a*1+b*0+c)*s,cudaDofVector2[neighbourEntities.template getEntityIndex< 1,  0 >()]);

}

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
void tnlNarrowBand< tnlGrid< 2,MeshReal, Device, MeshIndex >, Real, Index > :: setupSquare1101( Index i, Index j)
{
	tnlGridEntity< tnlGrid< 2,double, tnlHost, int >, 2, tnlGridEntityNoStencilStorage > Entity(Mesh);
	Entity.setCoordinates(CoordinatesType(i,j));
	Entity.refresh();
	tnlNeighbourGridEntityGetter<tnlGridEntity< MeshType, 2, tnlGridEntityNoStencilStorage >,2> neighbourEntities(Entity);
	Real al,be, a,b,c,s;
	al=abs(cudaDofVector[neighbourEntities.template getEntityIndex< 0,  1 >()]/
			(cudaDofVector[neighbourEntities.template getEntityIndex< 1,  1 >()]-
			 cudaDofVector[neighbourEntities.template getEntityIndex< 0,  1 >()]));

	be=abs(cudaDofVector[neighbourEntities.template getEntityIndex< 0,  1 >()]/
			(cudaDofVector[Entity.getIndex()]-
			 cudaDofVector[neighbourEntities.template getEntityIndex< 0,  1 >()]));

	a = be/al;
	b=1.0;
	c=-be;
	s= h/sqrt(a*a+b*b);


	cudaDofVector2[Entity.getIndex()]=fabsMin(abs(a*0+b*1+c)*s,cudaDofVector2[Entity.getIndex()]);
	cudaDofVector2[neighbourEntities.template getEntityIndex< 0,  1 >()]=fabsMin(abs(a*0+b*0+c)*s,cudaDofVector2[neighbourEntities.template getEntityIndex< 0,  1 >()]);
	cudaDofVector2[neighbourEntities.template getEntityIndex< 1,  1 >()]=fabsMin(abs(a*1+b*0+c)*s,cudaDofVector2[neighbourEntities.template getEntityIndex< 1,  1 >()]);
	cudaDofVector2[neighbourEntities.template getEntityIndex< 1,  0 >()]=fabsMin(-abs(a*1+b*1+c)*s,cudaDofVector2[neighbourEntities.template getEntityIndex< 1,  0 >()]);

}

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
void tnlNarrowBand< tnlGrid< 2,MeshReal, Device, MeshIndex >, Real, Index > :: setupSquare1011( Index i, Index j)
{
	tnlGridEntity< tnlGrid< 2,double, tnlHost, int >, 2, tnlGridEntityNoStencilStorage > Entity(Mesh);
	Entity.setCoordinates(CoordinatesType(i,j));
	Entity.refresh();
	tnlNeighbourGridEntityGetter<tnlGridEntity< MeshType, 2, tnlGridEntityNoStencilStorage >,2> neighbourEntities(Entity);
	Real al,be, a,b,c,s;
	al=abs(cudaDofVector[neighbourEntities.template getEntityIndex< 1,  0 >()]/
			(cudaDofVector[Entity.getIndex()]-
			 cudaDofVector[neighbourEntities.template getEntityIndex< 1,  0 >()]));

	be=abs(cudaDofVector[neighbourEntities.template getEntityIndex< 1,  0 >()]/
			(cudaDofVector[neighbourEntities.template getEntityIndex< 1,  1 >()]-
			 cudaDofVector[neighbourEntities.template getEntityIndex< 1,  0 >()]));

	a = be/al;
	b=1.0;
	c=-be;
	s= h/sqrt(a*a+b*b);


	cudaDofVector2[Entity.getIndex()]=fabsMin(abs(a*1+b*0+c)*s,cudaDofVector2[Entity.getIndex()]);
	cudaDofVector2[neighbourEntities.template getEntityIndex< 0,  1 >()]=fabsMin(-abs(a*1+b*1+c)*s,cudaDofVector2[neighbourEntities.template getEntityIndex< 0,  1 >()]);
	cudaDofVector2[neighbourEntities.template getEntityIndex< 1,  1 >()]=fabsMin(abs(a*0+b*1+c)*s,cudaDofVector2[neighbourEntities.template getEntityIndex< 1,  1 >()]);
	cudaDofVector2[neighbourEntities.template getEntityIndex< 1,  0 >()]=fabsMin(abs(a*0+b*0+c)*s,cudaDofVector2[neighbourEntities.template getEntityIndex< 1,  0 >()]);

}

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
void tnlNarrowBand< tnlGrid< 2,MeshReal, Device, MeshIndex >, Real, Index > :: setupSquare0111( Index i, Index j)
{
	tnlGridEntity< tnlGrid< 2,double, tnlHost, int >, 2, tnlGridEntityNoStencilStorage > Entity(Mesh);
	Entity.setCoordinates(CoordinatesType(i,j));
	Entity.refresh();
	tnlNeighbourGridEntityGetter<tnlGridEntity< MeshType, 2, tnlGridEntityNoStencilStorage >,2> neighbourEntities(Entity);
	Real al,be, a,b,c,s;
	al=abs(cudaDofVector[Entity.getIndex()]/
			(cudaDofVector[neighbourEntities.template getEntityIndex< 0,  1 >()]-
			 cudaDofVector[Entity.getIndex()]));

	be=abs(cudaDofVector[Entity.getIndex()]/
			(cudaDofVector[neighbourEntities.template getEntityIndex< 1,  0 >()]-
			 cudaDofVector[Entity.getIndex()]));

	a = be/al;
	b=1.0;
	c=-be;
	s= h/sqrt(a*a+b*b);


	cudaDofVector2[Entity.getIndex()]=fabsMin(-abs(a*0+b*0+c)*s,cudaDofVector2[Entity.getIndex()]);
	cudaDofVector2[neighbourEntities.template getEntityIndex< 0,  1 >()]=fabsMin(abs(a*1+b*0+c)*s,cudaDofVector2[neighbourEntities.template getEntityIndex< 0,  1 >()]);
	cudaDofVector2[neighbourEntities.template getEntityIndex< 1,  1 >()]=fabsMin(abs(a*1+b*1+c)*s,cudaDofVector2[neighbourEntities.template getEntityIndex< 1,  1 >()]);
	cudaDofVector2[neighbourEntities.template getEntityIndex< 1,  0 >()]=fabsMin(abs(a*0+b*1+c)*s,cudaDofVector2[neighbourEntities.template getEntityIndex< 1,  0 >()]);

}


template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
void tnlNarrowBand< tnlGrid< 2,MeshReal, Device, MeshIndex >, Real, Index > :: setupSquare0001( Index i, Index j)
{
	tnlGridEntity< tnlGrid< 2,double, tnlHost, int >, 2, tnlGridEntityNoStencilStorage > Entity(Mesh);
	Entity.setCoordinates(CoordinatesType(i,j));
	Entity.refresh();
	tnlNeighbourGridEntityGetter<tnlGridEntity< MeshType, 2, tnlGridEntityNoStencilStorage >,2> neighbourEntities(Entity);
	Real al,be, a,b,c,s;
	al=abs(cudaDofVector[neighbourEntities.template getEntityIndex< 1,  1 >()]/
			(cudaDofVector[neighbourEntities.template getEntityIndex< 1,  0 >()]-
			 cudaDofVector[neighbourEntities.template getEntityIndex< 1,  1 >()]));

	be=abs(cudaDofVector[neighbourEntities.template getEntityIndex< 1,  1 >()]/
			(cudaDofVector[neighbourEntities.template getEntityIndex< 0,  1 >()]-
			 cudaDofVector[neighbourEntities.template getEntityIndex< 1,  1 >()]));

	a = be/al;
	b=1.0;
	c=-be;
	s= h/sqrt(a*a+b*b);


	cudaDofVector2[Entity.getIndex()]=fabsMin(-abs(a*1+b*1+c)*s,cudaDofVector2[Entity.getIndex()]);
	cudaDofVector2[neighbourEntities.template getEntityIndex< 0,  1 >()]=fabsMin(-abs(a*0+b*1+c)*s,cudaDofVector2[neighbourEntities.template getEntityIndex< 0,  1 >()]);
	cudaDofVector2[neighbourEntities.template getEntityIndex< 1,  1 >()]=fabsMin(abs(a*0+b*0+c)*s,cudaDofVector2[neighbourEntities.template getEntityIndex< 1,  1 >()]);
	cudaDofVector2[neighbourEntities.template getEntityIndex< 1,  0 >()]=fabsMin(-abs(a*1+b*0+c)*s,cudaDofVector2[neighbourEntities.template getEntityIndex< 1,  0 >()]);

}

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
void tnlNarrowBand< tnlGrid< 2,MeshReal, Device, MeshIndex >, Real, Index > :: setupSquare0010( Index i, Index j)
{
	tnlGridEntity< tnlGrid< 2,double, tnlHost, int >, 2, tnlGridEntityNoStencilStorage > Entity(Mesh);
	Entity.setCoordinates(CoordinatesType(i,j));
	Entity.refresh();
	tnlNeighbourGridEntityGetter<tnlGridEntity< MeshType, 2, tnlGridEntityNoStencilStorage >,2> neighbourEntities(Entity);
	Real al,be, a,b,c,s;
	al=abs(cudaDofVector[neighbourEntities.template getEntityIndex< 0,  1 >()]/
			(cudaDofVector[neighbourEntities.template getEntityIndex< 1,  1 >()]-
			 cudaDofVector[neighbourEntities.template getEntityIndex< 0,  1 >()]));

	be=abs(cudaDofVector[neighbourEntities.template getEntityIndex< 0,  1 >()]/
			(cudaDofVector[Entity.getIndex()]-
			 cudaDofVector[neighbourEntities.template getEntityIndex< 0,  1 >()]));

	a = be/al;
	b=1.0;
	c=-be;
	s= h/sqrt(a*a+b*b);


	cudaDofVector2[Entity.getIndex()]=fabsMin(-abs(a*0+b*1+c)*s,cudaDofVector2[Entity.getIndex()]);
	cudaDofVector2[neighbourEntities.template getEntityIndex< 0,  1 >()]=fabsMin(-abs(a*0+b*0+c)*s,cudaDofVector2[neighbourEntities.template getEntityIndex< 0,  1 >()]);
	cudaDofVector2[neighbourEntities.template getEntityIndex< 1,  1 >()]=fabsMin(-abs(a*1+b*0+c)*s,cudaDofVector2[neighbourEntities.template getEntityIndex< 1,  1 >()]);
	cudaDofVector2[neighbourEntities.template getEntityIndex< 1,  0 >()]=fabsMin(abs(a*1+b*1+c)*s,cudaDofVector2[neighbourEntities.template getEntityIndex< 1,  0 >()]);

}

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
void tnlNarrowBand< tnlGrid< 2,MeshReal, Device, MeshIndex >, Real, Index > :: setupSquare0100( Index i, Index j)
{
	tnlGridEntity< tnlGrid< 2,double, tnlHost, int >, 2, tnlGridEntityNoStencilStorage > Entity(Mesh);
	Entity.setCoordinates(CoordinatesType(i,j));
	Entity.refresh();
	tnlNeighbourGridEntityGetter<tnlGridEntity< MeshType, 2, tnlGridEntityNoStencilStorage >,2> neighbourEntities(Entity);
	Real al,be, a,b,c,s;
	al=abs(cudaDofVector[neighbourEntities.template getEntityIndex< 1,  0 >()]/
			(cudaDofVector[Entity.getIndex()]-
			 cudaDofVector[neighbourEntities.template getEntityIndex< 1,  0 >()]));

	be=abs(cudaDofVector[neighbourEntities.template getEntityIndex< 1,  0 >()]/
			(cudaDofVector[neighbourEntities.template getEntityIndex< 1,  1 >()]-
			 cudaDofVector[neighbourEntities.template getEntityIndex< 1,  0 >()]));

	a = be/al;
	b=1.0;
	c=-be;
	s= h/sqrt(a*a+b*b);


	cudaDofVector2[Entity.getIndex()]=fabsMin(-abs(a*1+b*0+c)*s,cudaDofVector2[Entity.getIndex()]);
	cudaDofVector2[neighbourEntities.template getEntityIndex< 0,  1 >()]=fabsMin(abs(a*1+b*1+c)*s,cudaDofVector2[neighbourEntities.template getEntityIndex< 0,  1 >()]);
	cudaDofVector2[neighbourEntities.template getEntityIndex< 1,  1 >()]=fabsMin(-abs(a*0+b*1+c)*s,cudaDofVector2[neighbourEntities.template getEntityIndex< 1,  1 >()]);
	cudaDofVector2[neighbourEntities.template getEntityIndex< 1,  0 >()]=fabsMin(-abs(a*0+b*0+c)*s,cudaDofVector2[neighbourEntities.template getEntityIndex< 1,  0 >()]);

}

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
void tnlNarrowBand< tnlGrid< 2,MeshReal, Device, MeshIndex >, Real, Index > :: setupSquare1000( Index i, Index j)
{
	tnlGridEntity< tnlGrid< 2,double, tnlHost, int >, 2, tnlGridEntityNoStencilStorage > Entity(Mesh);
	Entity.setCoordinates(CoordinatesType(i,j));
	Entity.refresh();
	tnlNeighbourGridEntityGetter<tnlGridEntity< MeshType, 2, tnlGridEntityNoStencilStorage >,2> neighbourEntities(Entity);
	Real al,be, a,b,c,s;
	al=abs(cudaDofVector[Entity.getIndex()]/
			(cudaDofVector[neighbourEntities.template getEntityIndex< 0,  1 >()]-
			 cudaDofVector[Entity.getIndex()]));

	be=abs(cudaDofVector[Entity.getIndex()]/
			(cudaDofVector[neighbourEntities.template getEntityIndex< 1,  0 >()]-
			 cudaDofVector[Entity.getIndex()]));

	a = be/al;
	b=1.0;
	c=-be;
	s= h/sqrt(a*a+b*b);


	cudaDofVector2[Entity.getIndex()]=fabsMin(abs(a*0+b*0+c)*s,cudaDofVector2[Entity.getIndex()]);
	cudaDofVector2[neighbourEntities.template getEntityIndex< 0,  1 >()]=fabsMin(-abs(a*1+b*0+c)*s,cudaDofVector2[neighbourEntities.template getEntityIndex< 0,  1 >()]);
	cudaDofVector2[neighbourEntities.template getEntityIndex< 1,  1 >()]=fabsMin(-abs(a*1+b*1+c)*s,cudaDofVector2[neighbourEntities.template getEntityIndex< 1,  1 >()]);
	cudaDofVector2[neighbourEntities.template getEntityIndex< 1,  0 >()]=fabsMin(-abs(a*0+b*1+c)*s,cudaDofVector2[neighbourEntities.template getEntityIndex< 1,  0 >()]);

}





template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
void tnlNarrowBand< tnlGrid< 2,MeshReal, Device, MeshIndex >, Real, Index > :: setupSquare1100( Index i, Index j)
{
	tnlGridEntity< tnlGrid< 2,double, tnlHost, int >, 2, tnlGridEntityNoStencilStorage > Entity(Mesh);
	Entity.setCoordinates(CoordinatesType(i,j));
	Entity.refresh();
	tnlNeighbourGridEntityGetter<tnlGridEntity< MeshType, 2, tnlGridEntityNoStencilStorage >,2> neighbourEntities(Entity);
	Real al,be, a,b,c,s;
	al=abs(cudaDofVector[Entity.getIndex()]/
			(cudaDofVector[neighbourEntities.template getEntityIndex< 0,  1 >()]-
			 cudaDofVector[Entity.getIndex()]));

	be=abs(cudaDofVector[neighbourEntities.template getEntityIndex< 1,  0 >()]/
			(cudaDofVector[neighbourEntities.template getEntityIndex< 1,  1 >()]-
			 cudaDofVector[neighbourEntities.template getEntityIndex< 1,  0 >()]));

	a = al-be;
	b=1.0;
	c=-al;
	s= h/sqrt(a*a+b*b);


	cudaDofVector2[Entity.getIndex()]=fabsMin(abs(a*0+b*0+c)*s,cudaDofVector2[Entity.getIndex()]);
	cudaDofVector2[neighbourEntities.template getEntityIndex< 0,  1 >()]=fabsMin(-abs(a*0+b*1+c)*s,cudaDofVector2[neighbourEntities.template getEntityIndex< 0,  1 >()]);
	cudaDofVector2[neighbourEntities.template getEntityIndex< 1,  1 >()]=fabsMin(-abs(a*1+b*1+c)*s,cudaDofVector2[neighbourEntities.template getEntityIndex< 1,  1 >()]);
	cudaDofVector2[neighbourEntities.template getEntityIndex< 1,  0 >()]=fabsMin(abs(a*1+b*0+c)*s,cudaDofVector2[neighbourEntities.template getEntityIndex< 1,  0 >()]);

}

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
void tnlNarrowBand< tnlGrid< 2,MeshReal, Device, MeshIndex >, Real, Index > :: setupSquare1010( Index i, Index j)
{
	tnlGridEntity< tnlGrid< 2,double, tnlHost, int >, 2, tnlGridEntityNoStencilStorage > Entity(Mesh);
	Entity.setCoordinates(CoordinatesType(i,j));
	Entity.refresh();
	tnlNeighbourGridEntityGetter<tnlGridEntity< MeshType, 2, tnlGridEntityNoStencilStorage >,2> neighbourEntities(Entity);
	Real al,be, a,b,c,s;
	al=abs(cudaDofVector[Entity.getIndex()]/
			(cudaDofVector[neighbourEntities.template getEntityIndex< 1,  0 >()]-
			 cudaDofVector[Entity.getIndex()]));

	be=abs(cudaDofVector[neighbourEntities.template getEntityIndex< 0,  1 >()]/
			(cudaDofVector[neighbourEntities.template getEntityIndex< 1,  1 >()]-
			 cudaDofVector[neighbourEntities.template getEntityIndex< 0,  1 >()]));

	a = al-be;
	b=1.0;
	c=-be;
	s= h/sqrt(a*a+b*b);


	cudaDofVector2[Entity.getIndex()]=fabsMin(abs(a*0+b*0+c)*s,cudaDofVector2[Entity.getIndex()]);
	cudaDofVector2[neighbourEntities.template getEntityIndex< 0,  1 >()]=fabsMin(abs(a*1+b*0+c)*s,cudaDofVector2[neighbourEntities.template getEntityIndex< 0,  1 >()]);
	cudaDofVector2[neighbourEntities.template getEntityIndex< 1,  1 >()]=fabsMin(-abs(a*1+b*1+c)*s,cudaDofVector2[neighbourEntities.template getEntityIndex< 1,  1 >()]);
	cudaDofVector2[neighbourEntities.template getEntityIndex< 1,  0 >()]=fabsMin(-abs(a*0+b*1+c)*s,cudaDofVector2[neighbourEntities.template getEntityIndex< 1,  0 >()]);

}

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
void tnlNarrowBand< tnlGrid< 2,MeshReal, Device, MeshIndex >, Real, Index > :: setupSquare1001( Index i, Index j)
{
	tnlGridEntity< tnlGrid< 2,double, tnlHost, int >, 2, tnlGridEntityNoStencilStorage > Entity(Mesh);
	Entity.setCoordinates(CoordinatesType(i,j));
	Entity.refresh();
	tnlNeighbourGridEntityGetter<tnlGridEntity< MeshType, 2, tnlGridEntityNoStencilStorage >,2> neighbourEntities(Entity);
	cudaDofVector2[Entity.getIndex()]=fabsMin(cudaDofVector[Entity.getIndex()],cudaDofVector2[Entity.getIndex()]);
	cudaDofVector2[neighbourEntities.template getEntityIndex< 0,  1 >()]=fabsMin(cudaDofVector[neighbourEntities.template getEntityIndex< 0,  1 >()],cudaDofVector2[neighbourEntities.template getEntityIndex< 0,  1 >()]);
	cudaDofVector2[neighbourEntities.template getEntityIndex< 1,  1 >()]=fabsMin(cudaDofVector[neighbourEntities.template getEntityIndex< 1,  1 >()],cudaDofVector2[neighbourEntities.template getEntityIndex< 1,  1 >()]);
	cudaDofVector2[neighbourEntities.template getEntityIndex< 1,  0 >()]=fabsMin(cudaDofVector[neighbourEntities.template getEntityIndex< 1,  0 >()],cudaDofVector2[neighbourEntities.template getEntityIndex< 1,  0 >()]);

}







template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
void tnlNarrowBand< tnlGrid< 2,MeshReal, Device, MeshIndex >, Real, Index > :: setupSquare0011( Index i, Index j)
{
	tnlGridEntity< tnlGrid< 2,double, tnlHost, int >, 2, tnlGridEntityNoStencilStorage > Entity(Mesh);
	Entity.setCoordinates(CoordinatesType(i,j));
	Entity.refresh();
	tnlNeighbourGridEntityGetter<tnlGridEntity< MeshType, 2, tnlGridEntityNoStencilStorage >,2> neighbourEntities(Entity);
	Real al,be, a,b,c,s;
	al=abs(cudaDofVector[Entity.getIndex()]/
			(cudaDofVector[neighbourEntities.template getEntityIndex< 0,  1 >()]-
			 cudaDofVector[Entity.getIndex()]));

	be=abs(cudaDofVector[neighbourEntities.template getEntityIndex< 1,  0 >()]/
			(cudaDofVector[neighbourEntities.template getEntityIndex< 1,  1 >()]-
			 cudaDofVector[neighbourEntities.template getEntityIndex< 1,  0 >()]));

	a = al-be;
	b=1.0;
	c=-al;
	s= h/sqrt(a*a+b*b);


	cudaDofVector2[Entity.getIndex()]=fabsMin(-abs(a*0+b*0+c)*s,cudaDofVector2[Entity.getIndex()]);
	cudaDofVector2[neighbourEntities.template getEntityIndex< 0,  1 >()]=fabsMin(abs(a*0+b*1+c)*s,cudaDofVector2[neighbourEntities.template getEntityIndex< 0,  1 >()]);
	cudaDofVector2[neighbourEntities.template getEntityIndex< 1,  1 >()]=fabsMin(abs(a*1+b*1+c)*s,cudaDofVector2[neighbourEntities.template getEntityIndex< 1,  1 >()]);
	cudaDofVector2[neighbourEntities.template getEntityIndex< 1,  0 >()]=fabsMin(-abs(a*1+b*0+c)*s,cudaDofVector2[neighbourEntities.template getEntityIndex< 1,  0 >()]);

}

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
void tnlNarrowBand< tnlGrid< 2,MeshReal, Device, MeshIndex >, Real, Index > :: setupSquare0101( Index i, Index j)
{
	tnlGridEntity< tnlGrid< 2,double, tnlHost, int >, 2, tnlGridEntityNoStencilStorage > Entity(Mesh);
	Entity.setCoordinates(CoordinatesType(i,j));
	Entity.refresh();
	tnlNeighbourGridEntityGetter<tnlGridEntity< MeshType, 2, tnlGridEntityNoStencilStorage >,2> neighbourEntities(Entity);
	Real al,be, a,b,c,s;
	al=abs(cudaDofVector[Entity.getIndex()]/
			(cudaDofVector[neighbourEntities.template getEntityIndex< 1,  0 >()]-
			 cudaDofVector[Entity.getIndex()]));

	be=abs(cudaDofVector[neighbourEntities.template getEntityIndex< 0,  1 >()]/
			(cudaDofVector[neighbourEntities.template getEntityIndex< 1,  1 >()]-
			 cudaDofVector[neighbourEntities.template getEntityIndex< 0,  1 >()]));

	a = al-be;
	b=1.0;
	c=-be;
	s= h/sqrt(a*a+b*b);


	cudaDofVector2[Entity.getIndex()]=fabsMin(-abs(a*0+b*0+c)*s,cudaDofVector2[Entity.getIndex()]);
	cudaDofVector2[neighbourEntities.template getEntityIndex< 0,  1 >()]=fabsMin(-abs(a*1+b*0+c)*s,cudaDofVector2[neighbourEntities.template getEntityIndex< 0,  1 >()]);
	cudaDofVector2[neighbourEntities.template getEntityIndex< 1,  1 >()]=fabsMin(abs(a*1+b*1+c)*s,cudaDofVector2[neighbourEntities.template getEntityIndex< 1,  1 >()]);
	cudaDofVector2[neighbourEntities.template getEntityIndex< 1,  0 >()]=fabsMin(abs(a*0+b*1+c)*s,cudaDofVector2[neighbourEntities.template getEntityIndex< 1,  0 >()]);

}

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
void tnlNarrowBand< tnlGrid< 2,MeshReal, Device, MeshIndex >, Real, Index > :: setupSquare0110( Index i, Index j)
{
	tnlGridEntity< tnlGrid< 2,double, tnlHost, int >, 2, tnlGridEntityNoStencilStorage > Entity(Mesh);
	Entity.setCoordinates(CoordinatesType(i,j));
	Entity.refresh();
	tnlNeighbourGridEntityGetter<tnlGridEntity< MeshType, 2, tnlGridEntityNoStencilStorage >,2> neighbourEntities(Entity);
	cudaDofVector2[Entity.getIndex()]=fabsMin(cudaDofVector[Entity.getIndex()],cudaDofVector2[Entity.getIndex()]);
	cudaDofVector2[neighbourEntities.template getEntityIndex< 0,  1 >()]=fabsMin(cudaDofVector[neighbourEntities.template getEntityIndex< 0,  1 >()],cudaDofVector2[neighbourEntities.template getEntityIndex< 0,  1 >()]);
	cudaDofVector2[neighbourEntities.template getEntityIndex< 1,  1 >()]=fabsMin(cudaDofVector[neighbourEntities.template getEntityIndex< 1,  1 >()],cudaDofVector2[neighbourEntities.template getEntityIndex< 1,  1 >()]);
	cudaDofVector2[neighbourEntities.template getEntityIndex< 1,  0 >()]=fabsMin(cudaDofVector[neighbourEntities.template getEntityIndex< 1,  0 >()],cudaDofVector2[neighbourEntities.template getEntityIndex< 1,  0 >()]);
}
#endif




#endif /* TNLNARROWBAND_IMPL_H_ */
