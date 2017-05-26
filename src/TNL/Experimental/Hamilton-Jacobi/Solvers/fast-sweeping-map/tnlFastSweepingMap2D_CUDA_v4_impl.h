/***************************************************************************
                          tnlFastSweepingMap2D_CUDA_v4_impl.h  -  description
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

#include "tnlFastSweepingMap.h"

#define MAP_SOLVER_MAX_VALUE 3

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
String tnlFastSweepingMap< tnlGrid< 2,MeshReal, Device, MeshIndex >, Real, Index > :: getType()
{
	   return String( "tnlFastSweepingMap< " ) +
	          MeshType::getType() + ", " +
	          ::getType< Real >() + ", " +
	          ::getType< Index >() + " >";
}

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
tnlFastSweepingMap< tnlGrid< 2,MeshReal, Device, MeshIndex >, Real, Index > :: tnlFastSweepingMap()
:dofVector(Mesh)
{
}


template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
bool tnlFastSweepingMap< tnlGrid< 2,MeshReal, Device, MeshIndex >, Real, Index > :: init( const Config::ParameterContainer& parameters )
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

	const String& mapFile = parameters.getParameter <String>("map");
	if(! this->map.load( mapFile ))
		cout << "Failed to load map file : " << mapFile << endl;

	h = Mesh.template getSpaceStepsProducts< 1, 0 >();
	//Entity.refresh();
	counter = 0;

	const String& exact_input = parameters.getParameter< String >( "exact-input" );

	if(exact_input == "no")
		exactInput=false;
	else
		exactInput=true;


#ifdef HAVE_CUDA

	cudaMalloc(&(cudaDofVector), this->dofVector.getData().getSize()*sizeof(double));
	cudaMemcpy(cudaDofVector, this->dofVector.getData().getData(), this->dofVector.getData().getSize()*sizeof(double), cudaMemcpyHostToDevice);

	cudaMalloc(&(cudaDofVector2), this->dofVector.getData().getSize()*sizeof(double));
	cudaMemcpy(cudaDofVector2, this->dofVector.getData().getData(), this->dofVector.getData().getSize()*sizeof(double), cudaMemcpyHostToDevice);

	cudaMalloc(&(map_cuda), this->map.getSize()*sizeof(double));
	cudaMemcpy(map_cuda, this->map.getData(), this->map.getSize()*sizeof(double), cudaMemcpyHostToDevice);

	cudaMalloc(&(changed), sizeof(int));
	//counter == 0 --> setting changed to 0
	cudaMemcpy(changed, &counter, sizeof(int), cudaMemcpyHostToDevice);


	cudaMalloc(&(this->cudaSolver), sizeof(tnlFastSweepingMap< tnlGrid< 2,MeshReal, Device, MeshIndex >, Real, Index >));
	cudaMemcpy(this->cudaSolver, this,sizeof(tnlFastSweepingMap< tnlGrid< 2,MeshReal, Device, MeshIndex >, Real, Index >), cudaMemcpyHostToDevice);

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
bool tnlFastSweepingMap< tnlGrid< 2,MeshReal, Device, MeshIndex >, Real, Index > :: run()
{

	int n = Mesh.getDimensions().x();
	dim3 threadsPerBlock(1, 1024);
	dim3 numBlocks(4,1);

	int run = 1;
	int zero = 0;
	int cntr = 0;

	while(run != 0)
	{
		cudaMemcpy(this->changed, &zero, sizeof(int), cudaMemcpyHostToDevice);
		cudaDeviceSynchronize();
		checkCudaDevice;

		runCUDA<<<numBlocks,threadsPerBlock>>>(this->cudaSolver,0,0, this->changed);
		cudaDeviceSynchronize();
		checkCudaDevice;

		cudaMemcpy(&run, this->changed,sizeof(int), cudaMemcpyDeviceToHost);
		cudaDeviceSynchronize();
		checkCudaDevice;
		cntr++;
		cout << "Finished set of sweeps #" << cntr << "           " << run << endl;
	}

	cudaDeviceSynchronize();
	checkCudaDevice;

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
void tnlFastSweepingMap< tnlGrid< 2,MeshReal, Device, MeshIndex >, Real, Index > :: updateValue( Index i, Index j, Index* something_changed)
{
	tnlGridEntity< tnlGrid< 2,double, tnlHost, int >, 2, tnlGridEntityNoStencilStorage > Entity(Mesh);
	Entity.setCoordinates(CoordinatesType(i,j));
	Entity.refresh();

	if(map_cuda[Entity.getIndex()] != 0.0)
	{
		tnlNeighbourGridEntityGetter<tnlGridEntity< MeshType, 2, tnlGridEntityNoStencilStorage >,2> neighbourEntities(Entity);
		Real value = cudaDofVector2[Entity.getIndex()];
		Real im = abs(1.0/map_cuda[Entity.getIndex()]);
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


		if(abs(a-b) >= im*h)
			tmp = fabsMin(a,b) + sign(value)*im*h;
		else
			tmp = 0.5 * (a + b + sign(value)*sqrt(2.0 * im * h * im * h - (a - b) * (a - b) ) );

	//	cudaDofVector2[Entity.getIndex()]  = fabsMin(value, tmp);
		atomicFabsMin(&(cudaDofVector2[Entity.getIndex()]), tmp);

		if(abs(value)-abs(tmp) > 0.0)
			atomicMax(something_changed,1);
	}
	else
	{
		atomicFabsMin(&(cudaDofVector2[Entity.getIndex()]), MAP_SOLVER_MAX_VALUE);
	}

}


template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
__device__
bool tnlFastSweepingMap< tnlGrid< 2,MeshReal, Device, MeshIndex >, Real, Index > :: initGrid()
{
	int i = threadIdx.x + blockDim.x*blockIdx.x;
	int j = blockDim.y*blockIdx.y + threadIdx.y;

	tnlGridEntity< tnlGrid< 2,double, tnlHost, int >, 2, tnlGridEntityNoStencilStorage > Entity(Mesh);

	Entity.setCoordinates(CoordinatesType(i,j));
	Entity.refresh();
	tnlNeighbourGridEntityGetter<tnlGridEntity< MeshType, 2, tnlGridEntityNoStencilStorage >,2> neighbourEntities(Entity);

	int gid = Entity.getIndex();

	cudaDofVector2[gid] = INT_MAX*sign(cudaDofVector[gid]);

	if(abs(cudaDofVector[gid]) < 1.01*h)
	{
		cudaDofVector2[gid] = cudaDofVector[gid];
		if(map_cuda[gid] != 0.0)
			cudaDofVector2[gid] /=map_cuda[gid];
	}





//	if(i+1 < Mesh.getDimensions().x() && j+1 < Mesh.getDimensions().y() )
//	{
//		if(cudaDofVector[Entity.getIndex()] > 0)
//		{
//			if(cudaDofVector[neighbourEntities.template getEntityIndex< 1,  0 >()] > 0)
//			{
//				if(cudaDofVector[neighbourEntities.template getEntityIndex< 0,  1 >()] > 0)
//				{
//					if(cudaDofVector[neighbourEntities.template getEntityIndex< 1,  1 >()] > 0)
//						setupSquare1111(i,j);
//					else
//						setupSquare1110(i,j);
//				}
//				else
//				{
//					if(cudaDofVector[neighbourEntities.template getEntityIndex< 1,  1 >()] > 0)
//						setupSquare1101(i,j);
//					else
//						setupSquare1100(i,j);
//				}
//			}
//			else
//			{
//				if(cudaDofVector[neighbourEntities.template getEntityIndex< 0,  1 >()] > 0)
//				{
//					if(cudaDofVector[neighbourEntities.template getEntityIndex< 1,  1 >()] > 0)
//						setupSquare1011(i,j);
//					else
//						setupSquare1010(i,j);
//				}
//				else
//				{
//					if(cudaDofVector[neighbourEntities.template getEntityIndex< 1,  1 >()] > 0)
//						setupSquare1001(i,j);
//					else
//						setupSquare1000(i,j);
//				}
//			}
//		}
//		else
//		{
//			if(cudaDofVector[neighbourEntities.template getEntityIndex< 1,  0 >()] > 0)
//			{
//				if(cudaDofVector[neighbourEntities.template getEntityIndex< 0,  1 >()] > 0)
//				{
//					if(cudaDofVector[neighbourEntities.template getEntityIndex< 1,  1 >()] > 0)
//						setupSquare0111(i,j);
//					else
//						setupSquare0110(i,j);
//				}
//				else
//				{
//					if(cudaDofVector[neighbourEntities.template getEntityIndex< 1,  1 >()] > 0)
//						setupSquare0101(i,j);
//					else
//						setupSquare0100(i,j);
//				}
//			}
//			else
//			{
//				if(cudaDofVector[neighbourEntities.template getEntityIndex< 0,  1 >()] > 0)
//				{
//					if(cudaDofVector[neighbourEntities.template getEntityIndex< 1,  1 >()] > 0)
//						setupSquare0011(i,j);
//					else
//						setupSquare0010(i,j);
//				}
//				else
//				{
//					if(cudaDofVector[neighbourEntities.template getEntityIndex< 1,  1 >()] > 0)
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
Real tnlFastSweepingMap< tnlGrid< 2,MeshReal, Device, MeshIndex >, Real, Index > :: fabsMin( Real x, Real y)
{
	Real fx = abs(x);
	//Real fy = abs(y);

	//Real tmpMin = Min(fx,abs(y));

	if(Min(fx,abs(y)) == fx)
		return x;
	else
		return y;


}



__global__ void runCUDA(tnlFastSweepingMap< tnlGrid< 2,double, tnlHost, int >, double, int >* solver, int sweep, int i, int* changed)
{

	__shared__ int something_changed;
	if(threadIdx.x+threadIdx.y == 0)
		something_changed = 0;

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

	__syncthreads();
	if(blockIdx.x==0)
	{
		for(int k = 0; k < n*blockCount + blockDim.y; k++)
		{
			if(threadIdx.y  < k+1 && gy < n)
			{
				solver->updateValue(gx,gy,&something_changed);
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
				solver->updateValue(gx,gy,&something_changed);
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
				solver->updateValue(gx,gy,&something_changed);
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
				solver->updateValue(gx,gy,&something_changed);
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


	if(threadIdx.x+threadIdx.y == 0)
		atomicMax(changed, something_changed);




}


__global__ void initCUDA(tnlFastSweepingMap< tnlGrid< 2,double, tnlHost, int >, double, int >* solver)
{


	int gx = threadIdx.x + blockDim.x*blockIdx.x;
	int gy = blockDim.y*blockIdx.y + threadIdx.y;


	if(solver->Mesh.getDimensions().x() > gx && solver->Mesh.getDimensions().y() > gy)
	{
		solver->initGrid();
	}


}





































template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
void tnlFastSweepingMap< tnlGrid< 2,MeshReal, Device, MeshIndex >, Real, Index > :: setupSquare1111( Index i, Index j)
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
void tnlFastSweepingMap< tnlGrid< 2,MeshReal, Device, MeshIndex >, Real, Index > :: setupSquare0000( Index i, Index j)
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
void tnlFastSweepingMap< tnlGrid< 2,MeshReal, Device, MeshIndex >, Real, Index > :: setupSquare1110( Index i, Index j)
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
void tnlFastSweepingMap< tnlGrid< 2,MeshReal, Device, MeshIndex >, Real, Index > :: setupSquare1101( Index i, Index j)
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
void tnlFastSweepingMap< tnlGrid< 2,MeshReal, Device, MeshIndex >, Real, Index > :: setupSquare1011( Index i, Index j)
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
void tnlFastSweepingMap< tnlGrid< 2,MeshReal, Device, MeshIndex >, Real, Index > :: setupSquare0111( Index i, Index j)
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
void tnlFastSweepingMap< tnlGrid< 2,MeshReal, Device, MeshIndex >, Real, Index > :: setupSquare0001( Index i, Index j)
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
void tnlFastSweepingMap< tnlGrid< 2,MeshReal, Device, MeshIndex >, Real, Index > :: setupSquare0010( Index i, Index j)
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
void tnlFastSweepingMap< tnlGrid< 2,MeshReal, Device, MeshIndex >, Real, Index > :: setupSquare0100( Index i, Index j)
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
void tnlFastSweepingMap< tnlGrid< 2,MeshReal, Device, MeshIndex >, Real, Index > :: setupSquare1000( Index i, Index j)
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
void tnlFastSweepingMap< tnlGrid< 2,MeshReal, Device, MeshIndex >, Real, Index > :: setupSquare1100( Index i, Index j)
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
void tnlFastSweepingMap< tnlGrid< 2,MeshReal, Device, MeshIndex >, Real, Index > :: setupSquare1010( Index i, Index j)
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
void tnlFastSweepingMap< tnlGrid< 2,MeshReal, Device, MeshIndex >, Real, Index > :: setupSquare1001( Index i, Index j)
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
void tnlFastSweepingMap< tnlGrid< 2,MeshReal, Device, MeshIndex >, Real, Index > :: setupSquare0011( Index i, Index j)
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
void tnlFastSweepingMap< tnlGrid< 2,MeshReal, Device, MeshIndex >, Real, Index > :: setupSquare0101( Index i, Index j)
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
void tnlFastSweepingMap< tnlGrid< 2,MeshReal, Device, MeshIndex >, Real, Index > :: setupSquare0110( Index i, Index j)
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




#endif /* TNLFASTSWEEPING_IMPL_H_ */