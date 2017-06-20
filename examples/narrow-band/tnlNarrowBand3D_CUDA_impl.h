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
#ifndef TNLNARROWBAND3D_IMPL_H_
#define TNLNARROWBAND3D_IMPL_H_

#include "tnlNarrowBand.h"

//__device__
//double fabsMin( double x, double y)
//{
//	double fx = abs(x);
//
//	if(Min(fx,abs(y)) == fx)
//		return x;
//	else
//		return y;
//}
//
//__device__
//double atomicFabsMin(double* address, double val)
//{
//	unsigned long long int* address_as_ull =
//						  (unsigned long long int*)address;
//	unsigned long long int old = *address_as_ull, assumed;
//	do {
//		assumed = old;
//			old = atomicCAS(address_as_ull, assumed,__double_as_longlong( fabsMin(assumed,val) ));
//	} while (assumed != old);
//	return __longlong_as_double(old);
//}

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
String tnlNarrowBand< tnlGrid< 3,MeshReal, Device, MeshIndex >, Real, Index > :: getType()
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
bool tnlNarrowBand< tnlGrid< 3,MeshReal, Device, MeshIndex >, Real, Index > :: init( const Config::ParameterContainer& parameters )
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

	this->h = Mesh.template getSpaceStepsProducts< 1, 0, 0 >();
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


	cudaMalloc(&(this->cudaSolver), sizeof(tnlNarrowBand< tnlGrid< 3,MeshReal, Device, MeshIndex >, Real, Index >));
	cudaMemcpy(this->cudaSolver, this,sizeof(tnlNarrowBand< tnlGrid< 3,MeshReal, Device, MeshIndex >, Real, Index >), cudaMemcpyHostToDevice);

#endif

	int n = Mesh.getDimensions().x();
	dim3 threadsPerBlock(8, 8,8);
	dim3 numBlocks(n/8 + 1, n/8 +1, n/8 +1);

	cudaDeviceSynchronize();
	TNL_CHECK_CUDA_DEVICE;
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
bool tnlNarrowBand< tnlGrid< 3,MeshReal, Device, MeshIndex >, Real, Index > :: run()
{

	int n = Mesh.getDimensions().x();
	dim3 threadsPerBlock(1, 512);
	dim3 numBlocks(8,1);


	runCUDA<<<numBlocks,threadsPerBlock>>>(this->cudaSolver,0,0);

	cudaDeviceSynchronize();
	TNL_CHECK_CUDA_DEVICE;

	cudaMemcpy(this->dofVector.getData().getData(), cudaDofVector2, this->dofVector.getData().getSize()*sizeof(double), cudaMemcpyDeviceToHost);
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
void tnlNarrowBand< tnlGrid< 3,MeshReal, Device, MeshIndex >, Real, Index > :: updateValue( Index i, Index j, Index k)
{
	tnlGridEntity< tnlGrid< 3,double, tnlHost, int >, 3, tnlGridEntityNoStencilStorage > Entity(Mesh);
	Entity.setCoordinates(CoordinatesType(i,j,k));
	Entity.refresh();
	tnlNeighbourGridEntityGetter<tnlGridEntity< MeshType, 3, tnlGridEntityNoStencilStorage >,3> neighbourEntities(Entity);
	Real value = cudaDofVector2[Entity.getIndex()];
	Real a,b,c, tmp;

	if( i == 0 )
		a = cudaDofVector2[neighbourEntities.template getEntityIndex< 1,  0,  0 >()];
	else if( i == Mesh.getDimensions().x() - 1 )
		a = cudaDofVector2[neighbourEntities.template getEntityIndex< -1,  0,  0 >()];
	else
	{
		a = fabsMin( cudaDofVector2[neighbourEntities.template getEntityIndex< -1,  0,  0 >()],
				 cudaDofVector2[neighbourEntities.template getEntityIndex< 1,  0,  0 >()] );
	}

	if( j == 0 )
		b = cudaDofVector2[neighbourEntities.template getEntityIndex< 0,  1,  0 >()];
	else if( j == Mesh.getDimensions().y() - 1 )
		b = cudaDofVector2[neighbourEntities.template getEntityIndex< 0,  -1,  0 >()];
	else
	{
		b = fabsMin( cudaDofVector2[neighbourEntities.template getEntityIndex< 0,  -1,  0 >()],
				 cudaDofVector2[neighbourEntities.template getEntityIndex< 0,  1,  0 >()] );
	}

	if( k == 0 )
		c = cudaDofVector2[neighbourEntities.template getEntityIndex< 0,  0,  1 >()];
	else if( k == Mesh.getDimensions().z() - 1 )
		c = cudaDofVector2[neighbourEntities.template getEntityIndex< 0,  0,  -1 >()];
	else
	{
		c = fabsMin( cudaDofVector2[neighbourEntities.template getEntityIndex< 0,  0,  -1 >()],
				 cudaDofVector2[neighbourEntities.template getEntityIndex< 0,  0,  1 >()] );
	}

	Real hD = 3.0*h*h - 2.0*(a*a + b*b + c*c - a*b - a*c - b*c);

	if(hD < 0.0)
		tmp = fabsMin(a,fabsMin(b,c)) + sign(value)*h;
	else
		tmp = (1.0/3.0) * ( a + b + c + sign(value)*sqrt(hD) );

	atomicFabsMin(&cudaDofVector2[Entity.getIndex()],tmp);

}


template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
__device__
bool tnlNarrowBand< tnlGrid< 3,MeshReal, Device, MeshIndex >, Real, Index > :: initGrid(int i, int j, int k)
{
	tnlGridEntity< tnlGrid< 3,double, tnlHost, int >, 3, tnlGridEntityNoStencilStorage > Entity(Mesh);
	Entity.setCoordinates(CoordinatesType(i,j,k));
	Entity.refresh();
	int gid = Entity.getIndex();

	if(abs(cudaDofVector[gid]) < 1.8*h)
		cudaDofVector2[gid] = cudaDofVector[gid];
	else
		cudaDofVector2[gid] = INT_MAX*sign(cudaDofVector[gid]);

	return true;
}


template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
__device__
Real tnlNarrowBand< tnlGrid< 3,MeshReal, Device, MeshIndex >, Real, Index > :: fabsMin( Real x, Real y)
{
	Real fx = abs(x);
	if(Min(fx,abs(y)) == fx)
		return x;
	else
		return y;


}



__global__ void runCUDA(tnlNarrowBand< tnlGrid< 3,double, tnlHost, int >, double, int >* solver, int sweep, int i)
{

	int gx = 0;
	int gy = threadIdx.y;

	int n = solver->Mesh.getDimensions().x();
	int blockCount = n/blockDim.y +1;

	if(blockIdx.x==0)
	{
		for(int gz = 0; gz < n;gz++)
		{
		gx = 0;
		gy = threadIdx.y;
		for(int k = 0; k < n*blockCount + blockDim.y; k++)
		{
			if(threadIdx.y  < k+1 && gy < n)
			{
				solver->updateValue(gx,gy,gz);
				gx++;
				if(gx==n)
				{
					gx=0;
					gy+=blockDim.y;
				}
			}


			__syncthreads();
		}
		__syncthreads();
		}
	}
	else if(blockIdx.x==1)
	{
		for(int gz = 0; gz < n;gz++)
		{
		gx=n-1;
		gy=threadIdx.y;

		for(int k = 0; k < n*blockCount + blockDim.y; k++)
		{
			if(threadIdx.y  < k+1 && gy < n)
			{
				solver->updateValue(gx,gy,gz);
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
	}
	else if(blockIdx.x==2)
	{

		for(int gz = 0; gz < n;gz++)
		{
		gx=0;
		gy=n-threadIdx.y-1;
		for(int k = 0; k < n*blockCount + blockDim.y; k++)
		{
			if(threadIdx.y  < k+1 && gy > -1)
			{
				solver->updateValue(gx,gy,gz);
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
	}
	else if(blockIdx.x==3)
	{
		for(int gz = 0; gz < n;gz++)
		{
		gx=n-1;
		gy=n-threadIdx.y-1;

		for(int k = 0; k < n*blockCount + blockDim.y; k++)
		{
			if(threadIdx.y  < k+1 && gy > -1)
			{
				solver->updateValue(gx,gy,gz);
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




	else if(blockIdx.x==4)
	{
		for(int gz = n-1; gz > -1;gz--)
		{
		gx = 0;
		gy = threadIdx.y;
		for(int k = 0; k < n*blockCount + blockDim.y; k++)
		{
			if(threadIdx.y  < k+1 && gy < n)
			{
				solver->updateValue(gx,gy,gz);
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
	}
	else if(blockIdx.x==5)
	{
		for(int gz = n-1; gz > -1;gz--)
		{
		gx=n-1;
		gy=threadIdx.y;

		for(int k = 0; k < n*blockCount + blockDim.y; k++)
		{
			if(threadIdx.y  < k+1 && gy < n)
			{
				solver->updateValue(gx,gy,gz);
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
	}
	else if(blockIdx.x==6)
	{

		for(int gz = n-1; gz > -1;gz--)
		{
		gx=0;
		gy=n-threadIdx.y-1;
		for(int k = 0; k < n*blockCount + blockDim.y; k++)
		{
			if(threadIdx.y  < k+1 && gy > -1)
			{
				solver->updateValue(gx,gy,gz);
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
	}
	else if(blockIdx.x==7)
	{
		for(int gz = n-1; gz > -1;gz--)
		{
		gx=n-1;
		gy=n-threadIdx.y-1;

		for(int k = 0; k < n*blockCount + blockDim.y; k++)
		{
			if(threadIdx.y  < k+1 && gy > -1)
			{
				solver->updateValue(gx,gy,gz);
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




}


__global__ void initCUDA(tnlNarrowBand< tnlGrid< 3,double, tnlHost, int >, double, int >* solver)
{
	int gx = threadIdx.x + blockDim.x*blockIdx.x;
	int gy = blockDim.y*blockIdx.y + threadIdx.y;
	int gz = blockDim.z*blockIdx.z + threadIdx.z;

	if(solver->Mesh.getDimensions().x() > gx && solver->Mesh.getDimensions().y() > gy && solver->Mesh.getDimensions().z() > gz)
	{
		solver->initGrid(gx,gy,gz);
	}


}



































//template< typename MeshReal,
//          typename Device,
//          typename MeshIndex,
//          typename Real,
//          typename Index >
//void tnlNarrowBand< tnlGrid< 3,MeshReal, Device, MeshIndex >, Real, Index > :: setupSquare1111( Index i, Index j)
//{
//	Index index = Mesh.getCellIndex(CoordinatesType(i,j));
//	cudaDofVector2[index]=fabsMin(INT_MAX,cudaDofVector2[(index)]);
//	cudaDofVector2[Mesh.template getCellNextToCell<0,1>(index)]=fabsMin(INT_MAX,cudaDofVector2[Mesh.template getCellNextToCell<0,1>(index)]);
//	cudaDofVector2[Mesh.template getCellNextToCell<1,1>(index)]=fabsMin(INT_MAX,cudaDofVector2[Mesh.template getCellNextToCell<1,1>(index)]);
//	cudaDofVector2[Mesh.template getCellNextToCell<1,0>(index)]=fabsMin(INT_MAX,cudaDofVector2[Mesh.template getCellNextToCell<1,0>(index)]);
//
//}
//
//
//template< typename MeshReal,
//          typename Device,
//          typename MeshIndex,
//          typename Real,
//          typename Index >
//void tnlNarrowBand< tnlGrid< 3,MeshReal, Device, MeshIndex >, Real, Index > :: setupSquare0000( Index i, Index j)
//{
//	Index index = Mesh.getCellIndex(CoordinatesType(i,j));
//	cudaDofVector2[index]=fabsMin(-INT_MAX,cudaDofVector2[(index)]);
//	cudaDofVector2[Mesh.template getCellNextToCell<0,1>(index)]=fabsMin(-INT_MAX,cudaDofVector2[Mesh.template getCellNextToCell<0,1>(index)]);
//	cudaDofVector2[Mesh.template getCellNextToCell<1,1>(index)]=fabsMin(-INT_MAX,cudaDofVector2[Mesh.template getCellNextToCell<1,1>(index)]);
//	cudaDofVector2[Mesh.template getCellNextToCell<1,0>(index)]=fabsMin(-INT_MAX,cudaDofVector2[Mesh.template getCellNextToCell<1,0>(index)]);
//
//}
//
//
//template< typename MeshReal,
//          typename Device,
//          typename MeshIndex,
//          typename Real,
//          typename Index >
//void tnlNarrowBand< tnlGrid< 3,MeshReal, Device, MeshIndex >, Real, Index > :: setupSquare1110( Index i, Index j)
//{
//	Index index = Mesh.getCellIndex(CoordinatesType(i,j));
//	Real al,be, a,b,c,s;
//	al=abs(cudaDofVector[Mesh.template getCellNextToCell<1,1>(index)]/
//			(cudaDofVector[Mesh.template getCellNextToCell<1,0>(index)]-
//			 cudaDofVector[Mesh.template getCellNextToCell<1,1>(index)]));
//
//	be=abs(cudaDofVector[Mesh.template getCellNextToCell<1,1>(index)]/
//			(cudaDofVector[Mesh.template getCellNextToCell<0,1>(index)]-
//			 cudaDofVector[Mesh.template getCellNextToCell<1,1>(index)]));
//
//	a = be/al;
//	b=1.0;
//	c=-be;
//	s= h/sqrt(a*a+b*b);
//
//
//	cudaDofVector2[index]=fabsMin(abs(a*1+b*1+c)*s,cudaDofVector2[(index)]);
//	cudaDofVector2[Mesh.template getCellNextToCell<0,1>(index)]=fabsMin(abs(a*0+b*1+c)*s,cudaDofVector2[Mesh.template getCellNextToCell<0,1>(index)]);
//	cudaDofVector2[Mesh.template getCellNextToCell<1,1>(index)]=fabsMin(-abs(a*0+b*0+c)*s,cudaDofVector2[Mesh.template getCellNextToCell<1,1>(index)]);
//	cudaDofVector2[Mesh.template getCellNextToCell<1,0>(index)]=fabsMin(abs(a*1+b*0+c)*s,cudaDofVector2[Mesh.template getCellNextToCell<1,0>(index)]);
//
//}
//
//template< typename MeshReal,
//          typename Device,
//          typename MeshIndex,
//          typename Real,
//          typename Index >
//void tnlNarrowBand< tnlGrid< 3,MeshReal, Device, MeshIndex >, Real, Index > :: setupSquare1101( Index i, Index j)
//{
//	Index index = Mesh.getCellIndex(CoordinatesType(i,j));
//	Real al,be, a,b,c,s;
//	al=abs(cudaDofVector[Mesh.template getCellNextToCell<0,1>(index)]/
//			(cudaDofVector[Mesh.template getCellNextToCell<1,1>(index)]-
//			 cudaDofVector[Mesh.template getCellNextToCell<0,1>(index)]));
//
//	be=abs(cudaDofVector[Mesh.template getCellNextToCell<0,1>(index)]/
//			(cudaDofVector[Mesh.template getCellNextToCell<0,0>(index)]-
//			 cudaDofVector[Mesh.template getCellNextToCell<0,1>(index)]));
//
//	a = be/al;
//	b=1.0;
//	c=-be;
//	s= h/sqrt(a*a+b*b);
//
//
//	cudaDofVector2[index]=fabsMin(abs(a*0+b*1+c)*s,cudaDofVector2[(index)]);
//	cudaDofVector2[Mesh.template getCellNextToCell<0,1>(index)]=fabsMin(abs(a*0+b*0+c)*s,cudaDofVector2[Mesh.template getCellNextToCell<0,1>(index)]);
//	cudaDofVector2[Mesh.template getCellNextToCell<1,1>(index)]=fabsMin(abs(a*1+b*0+c)*s,cudaDofVector2[Mesh.template getCellNextToCell<1,1>(index)]);
//	cudaDofVector2[Mesh.template getCellNextToCell<1,0>(index)]=fabsMin(-abs(a*1+b*1+c)*s,cudaDofVector2[Mesh.template getCellNextToCell<1,0>(index)]);
//
//}
//
//template< typename MeshReal,
//          typename Device,
//          typename MeshIndex,
//          typename Real,
//          typename Index >
//void tnlNarrowBand< tnlGrid< 3,MeshReal, Device, MeshIndex >, Real, Index > :: setupSquare1011( Index i, Index j)
//{
//	Index index = Mesh.getCellIndex(CoordinatesType(i,j));
//	Real al,be, a,b,c,s;
//	al=abs(cudaDofVector[Mesh.template getCellNextToCell<1,0>(index)]/
//			(cudaDofVector[Mesh.template getCellNextToCell<0,0>(index)]-
//			 cudaDofVector[Mesh.template getCellNextToCell<1,0>(index)]));
//
//	be=abs(cudaDofVector[Mesh.template getCellNextToCell<1,0>(index)]/
//			(cudaDofVector[Mesh.template getCellNextToCell<1,1>(index)]-
//			 cudaDofVector[Mesh.template getCellNextToCell<1,0>(index)]));
//
//	a = be/al;
//	b=1.0;
//	c=-be;
//	s= h/sqrt(a*a+b*b);
//
//
//	cudaDofVector2[index]=fabsMin(abs(a*1+b*0+c)*s,cudaDofVector2[(index)]);
//	cudaDofVector2[Mesh.template getCellNextToCell<0,1>(index)]=fabsMin(-abs(a*1+b*1+c)*s,cudaDofVector2[Mesh.template getCellNextToCell<0,1>(index)]);
//	cudaDofVector2[Mesh.template getCellNextToCell<1,1>(index)]=fabsMin(abs(a*0+b*1+c)*s,cudaDofVector2[Mesh.template getCellNextToCell<1,1>(index)]);
//	cudaDofVector2[Mesh.template getCellNextToCell<1,0>(index)]=fabsMin(abs(a*0+b*0+c)*s,cudaDofVector2[Mesh.template getCellNextToCell<1,0>(index)]);
//
//}
//
//template< typename MeshReal,
//          typename Device,
//          typename MeshIndex,
//          typename Real,
//          typename Index >
//void tnlNarrowBand< tnlGrid< 3,MeshReal, Device, MeshIndex >, Real, Index > :: setupSquare0111( Index i, Index j)
//{
//	Index index = Mesh.getCellIndex(CoordinatesType(i,j));
//	Real al,be, a,b,c,s;
//	al=abs(cudaDofVector[Mesh.template getCellNextToCell<0,0>(index)]/
//			(cudaDofVector[Mesh.template getCellNextToCell<0,1>(index)]-
//			 cudaDofVector[Mesh.template getCellNextToCell<0,0>(index)]));
//
//	be=abs(cudaDofVector[Mesh.template getCellNextToCell<0,0>(index)]/
//			(cudaDofVector[Mesh.template getCellNextToCell<1,0>(index)]-
//			 cudaDofVector[Mesh.template getCellNextToCell<0,0>(index)]));
//
//	a = be/al;
//	b=1.0;
//	c=-be;
//	s= h/sqrt(a*a+b*b);
//
//
//	cudaDofVector2[index]=fabsMin(-abs(a*0+b*0+c)*s,cudaDofVector2[(index)]);
//	cudaDofVector2[Mesh.template getCellNextToCell<0,1>(index)]=fabsMin(abs(a*1+b*0+c)*s,cudaDofVector2[Mesh.template getCellNextToCell<0,1>(index)]);
//	cudaDofVector2[Mesh.template getCellNextToCell<1,1>(index)]=fabsMin(abs(a*1+b*1+c)*s,cudaDofVector2[Mesh.template getCellNextToCell<1,1>(index)]);
//	cudaDofVector2[Mesh.template getCellNextToCell<1,0>(index)]=fabsMin(abs(a*0+b*1+c)*s,cudaDofVector2[Mesh.template getCellNextToCell<1,0>(index)]);
//
//}
//
//
//template< typename MeshReal,
//          typename Device,
//          typename MeshIndex,
//          typename Real,
//          typename Index >
//void tnlNarrowBand< tnlGrid< 3,MeshReal, Device, MeshIndex >, Real, Index > :: setupSquare0001( Index i, Index j)
//{
//	Index index = Mesh.getCellIndex(CoordinatesType(i,j));
//	Real al,be, a,b,c,s;
//	al=abs(cudaDofVector[Mesh.template getCellNextToCell<1,1>(index)]/
//			(cudaDofVector[Mesh.template getCellNextToCell<1,0>(index)]-
//			 cudaDofVector[Mesh.template getCellNextToCell<1,1>(index)]));
//
//	be=abs(cudaDofVector[Mesh.template getCellNextToCell<1,1>(index)]/
//			(cudaDofVector[Mesh.template getCellNextToCell<0,1>(index)]-
//			 cudaDofVector[Mesh.template getCellNextToCell<1,1>(index)]));
//
//	a = be/al;
//	b=1.0;
//	c=-be;
//	s= h/sqrt(a*a+b*b);
//
//
//	cudaDofVector2[index]=fabsMin(-abs(a*1+b*1+c)*s,cudaDofVector2[(index)]);
//	cudaDofVector2[Mesh.template getCellNextToCell<0,1>(index)]=fabsMin(-abs(a*0+b*1+c)*s,cudaDofVector2[Mesh.template getCellNextToCell<0,1>(index)]);
//	cudaDofVector2[Mesh.template getCellNextToCell<1,1>(index)]=fabsMin(abs(a*0+b*0+c)*s,cudaDofVector2[Mesh.template getCellNextToCell<1,1>(index)]);
//	cudaDofVector2[Mesh.template getCellNextToCell<1,0>(index)]=fabsMin(-abs(a*1+b*0+c)*s,cudaDofVector2[Mesh.template getCellNextToCell<1,0>(index)]);
//
//}
//
//template< typename MeshReal,
//          typename Device,
//          typename MeshIndex,
//          typename Real,
//          typename Index >
//void tnlNarrowBand< tnlGrid< 3,MeshReal, Device, MeshIndex >, Real, Index > :: setupSquare0010( Index i, Index j)
//{
//	Index index = Mesh.getCellIndex(CoordinatesType(i,j));
//	Real al,be, a,b,c,s;
//	al=abs(cudaDofVector[Mesh.template getCellNextToCell<0,1>(index)]/
//			(cudaDofVector[Mesh.template getCellNextToCell<1,1>(index)]-
//			 cudaDofVector[Mesh.template getCellNextToCell<0,1>(index)]));
//
//	be=abs(cudaDofVector[Mesh.template getCellNextToCell<0,1>(index)]/
//			(cudaDofVector[Mesh.template getCellNextToCell<0,0>(index)]-
//			 cudaDofVector[Mesh.template getCellNextToCell<0,1>(index)]));
//
//	a = be/al;
//	b=1.0;
//	c=-be;
//	s= h/sqrt(a*a+b*b);
//
//
//	cudaDofVector2[index]=fabsMin(-abs(a*0+b*1+c)*s,cudaDofVector2[(index)]);
//	cudaDofVector2[Mesh.template getCellNextToCell<0,1>(index)]=fabsMin(-abs(a*0+b*0+c)*s,cudaDofVector2[Mesh.template getCellNextToCell<0,1>(index)]);
//	cudaDofVector2[Mesh.template getCellNextToCell<1,1>(index)]=fabsMin(-abs(a*1+b*0+c)*s,cudaDofVector2[Mesh.template getCellNextToCell<1,1>(index)]);
//	cudaDofVector2[Mesh.template getCellNextToCell<1,0>(index)]=fabsMin(abs(a*1+b*1+c)*s,cudaDofVector2[Mesh.template getCellNextToCell<1,0>(index)]);
//
//}
//
//template< typename MeshReal,
//          typename Device,
//          typename MeshIndex,
//          typename Real,
//          typename Index >
//void tnlNarrowBand< tnlGrid< 3,MeshReal, Device, MeshIndex >, Real, Index > :: setupSquare0100( Index i, Index j)
//{
//	Index index = Mesh.getCellIndex(CoordinatesType(i,j));
//	Real al,be, a,b,c,s;
//	al=abs(cudaDofVector[Mesh.template getCellNextToCell<1,0>(index)]/
//			(cudaDofVector[Mesh.template getCellNextToCell<0,0>(index)]-
//			 cudaDofVector[Mesh.template getCellNextToCell<1,0>(index)]));
//
//	be=abs(cudaDofVector[Mesh.template getCellNextToCell<1,0>(index)]/
//			(cudaDofVector[Mesh.template getCellNextToCell<1,1>(index)]-
//			 cudaDofVector[Mesh.template getCellNextToCell<1,0>(index)]));
//
//	a = be/al;
//	b=1.0;
//	c=-be;
//	s= h/sqrt(a*a+b*b);
//
//
//	cudaDofVector2[index]=fabsMin(-abs(a*1+b*0+c)*s,cudaDofVector2[(index)]);
//	cudaDofVector2[Mesh.template getCellNextToCell<0,1>(index)]=fabsMin(abs(a*1+b*1+c)*s,cudaDofVector2[Mesh.template getCellNextToCell<0,1>(index)]);
//	cudaDofVector2[Mesh.template getCellNextToCell<1,1>(index)]=fabsMin(-abs(a*0+b*1+c)*s,cudaDofVector2[Mesh.template getCellNextToCell<1,1>(index)]);
//	cudaDofVector2[Mesh.template getCellNextToCell<1,0>(index)]=fabsMin(-abs(a*0+b*0+c)*s,cudaDofVector2[Mesh.template getCellNextToCell<1,0>(index)]);
//
//}
//
//template< typename MeshReal,
//          typename Device,
//          typename MeshIndex,
//          typename Real,
//          typename Index >
//void tnlNarrowBand< tnlGrid< 3,MeshReal, Device, MeshIndex >, Real, Index > :: setupSquare1000( Index i, Index j)
//{
//	Index index = Mesh.getCellIndex(CoordinatesType(i,j));
//	Real al,be, a,b,c,s;
//	al=abs(cudaDofVector[Mesh.template getCellNextToCell<0,0>(index)]/
//			(cudaDofVector[Mesh.template getCellNextToCell<0,1>(index)]-
//			 cudaDofVector[Mesh.template getCellNextToCell<0,0>(index)]));
//
//	be=abs(cudaDofVector[Mesh.template getCellNextToCell<0,0>(index)]/
//			(cudaDofVector[Mesh.template getCellNextToCell<1,0>(index)]-
//			 cudaDofVector[Mesh.template getCellNextToCell<0,0>(index)]));
//
//	a = be/al;
//	b=1.0;
//	c=-be;
//	s= h/sqrt(a*a+b*b);
//
//
//	cudaDofVector2[index]=fabsMin(abs(a*0+b*0+c)*s,cudaDofVector2[(index)]);
//	cudaDofVector2[Mesh.template getCellNextToCell<0,1>(index)]=fabsMin(-abs(a*1+b*0+c)*s,cudaDofVector2[Mesh.template getCellNextToCell<0,1>(index)]);
//	cudaDofVector2[Mesh.template getCellNextToCell<1,1>(index)]=fabsMin(-abs(a*1+b*1+c)*s,cudaDofVector2[Mesh.template getCellNextToCell<1,1>(index)]);
//	cudaDofVector2[Mesh.template getCellNextToCell<1,0>(index)]=fabsMin(-abs(a*0+b*1+c)*s,cudaDofVector2[Mesh.template getCellNextToCell<1,0>(index)]);
//
//}
//
//
//
//
//
//template< typename MeshReal,
//          typename Device,
//          typename MeshIndex,
//          typename Real,
//          typename Index >
//void tnlNarrowBand< tnlGrid< 3,MeshReal, Device, MeshIndex >, Real, Index > :: setupSquare1100( Index i, Index j)
//{
//	Index index = Mesh.getCellIndex(CoordinatesType(i,j));
//	Real al,be, a,b,c,s;
//	al=abs(cudaDofVector[Mesh.template getCellNextToCell<0,0>(index)]/
//			(cudaDofVector[Mesh.template getCellNextToCell<0,1>(index)]-
//			 cudaDofVector[Mesh.template getCellNextToCell<0,0>(index)]));
//
//	be=abs(cudaDofVector[Mesh.template getCellNextToCell<1,0>(index)]/
//			(cudaDofVector[Mesh.template getCellNextToCell<1,1>(index)]-
//			 cudaDofVector[Mesh.template getCellNextToCell<1,0>(index)]));
//
//	a = al-be;
//	b=1.0;
//	c=-al;
//	s= h/sqrt(a*a+b*b);
//
//
//	cudaDofVector2[index]=fabsMin(abs(a*0+b*0+c)*s,cudaDofVector2[(index)]);
//	cudaDofVector2[Mesh.template getCellNextToCell<0,1>(index)]=fabsMin(-abs(a*0+b*1+c)*s,cudaDofVector2[Mesh.template getCellNextToCell<0,1>(index)]);
//	cudaDofVector2[Mesh.template getCellNextToCell<1,1>(index)]=fabsMin(-abs(a*1+b*1+c)*s,cudaDofVector2[Mesh.template getCellNextToCell<1,1>(index)]);
//	cudaDofVector2[Mesh.template getCellNextToCell<1,0>(index)]=fabsMin(abs(a*1+b*0+c)*s,cudaDofVector2[Mesh.template getCellNextToCell<1,0>(index)]);
//
//}
//
//template< typename MeshReal,
//          typename Device,
//          typename MeshIndex,
//          typename Real,
//          typename Index >
//void tnlNarrowBand< tnlGrid< 3,MeshReal, Device, MeshIndex >, Real, Index > :: setupSquare1010( Index i, Index j)
//{
//	Index index = Mesh.getCellIndex(CoordinatesType(i,j));
//	Real al,be, a,b,c,s;
//	al=abs(cudaDofVector[Mesh.template getCellNextToCell<0,0>(index)]/
//			(cudaDofVector[Mesh.template getCellNextToCell<1,0>(index)]-
//			 cudaDofVector[Mesh.template getCellNextToCell<0,0>(index)]));
//
//	be=abs(cudaDofVector[Mesh.template getCellNextToCell<0,1>(index)]/
//			(cudaDofVector[Mesh.template getCellNextToCell<1,1>(index)]-
//			 cudaDofVector[Mesh.template getCellNextToCell<0,1>(index)]));
//
//	a = al-be;
//	b=1.0;
//	c=-be;
//	s= h/sqrt(a*a+b*b);
//
//
//	cudaDofVector2[index]=fabsMin(abs(a*0+b*0+c)*s,cudaDofVector2[(index)]);
//	cudaDofVector2[Mesh.template getCellNextToCell<0,1>(index)]=fabsMin(abs(a*1+b*0+c)*s,cudaDofVector2[Mesh.template getCellNextToCell<0,1>(index)]);
//	cudaDofVector2[Mesh.template getCellNextToCell<1,1>(index)]=fabsMin(-abs(a*1+b*1+c)*s,cudaDofVector2[Mesh.template getCellNextToCell<1,1>(index)]);
//	cudaDofVector2[Mesh.template getCellNextToCell<1,0>(index)]=fabsMin(-abs(a*0+b*1+c)*s,cudaDofVector2[Mesh.template getCellNextToCell<1,0>(index)]);
//
//}
//
//template< typename MeshReal,
//          typename Device,
//          typename MeshIndex,
//          typename Real,
//          typename Index >
//void tnlNarrowBand< tnlGrid< 3,MeshReal, Device, MeshIndex >, Real, Index > :: setupSquare1001( Index i, Index j)
//{
//	Index index = Mesh.getCellIndex(CoordinatesType(i,j));
//	cudaDofVector2[index]=fabsMin(cudaDofVector[index],cudaDofVector2[(index)]);
//	cudaDofVector2[Mesh.template getCellNextToCell<0,1>(index)]=fabsMin(cudaDofVector[Mesh.template getCellNextToCell<0,1>(index)],cudaDofVector2[Mesh.template getCellNextToCell<0,1>(index)]);
//	cudaDofVector2[Mesh.template getCellNextToCell<1,1>(index)]=fabsMin(cudaDofVector[Mesh.template getCellNextToCell<1,1>(index)],cudaDofVector2[Mesh.template getCellNextToCell<1,1>(index)]);
//	cudaDofVector2[Mesh.template getCellNextToCell<1,0>(index)]=fabsMin(cudaDofVector[Mesh.template getCellNextToCell<1,0>(index)],cudaDofVector2[Mesh.template getCellNextToCell<1,0>(index)]);
//
//}
//
//
//
//
//
//
//
//template< typename MeshReal,
//          typename Device,
//          typename MeshIndex,
//          typename Real,
//          typename Index >
//void tnlNarrowBand< tnlGrid< 3,MeshReal, Device, MeshIndex >, Real, Index > :: setupSquare0011( Index i, Index j)
//{
//	Index index = Mesh.getCellIndex(CoordinatesType(i,j));
//	Real al,be, a,b,c,s;
//	al=abs(cudaDofVector[Mesh.template getCellNextToCell<0,0>(index)]/
//			(cudaDofVector[Mesh.template getCellNextToCell<0,1>(index)]-
//			 cudaDofVector[Mesh.template getCellNextToCell<0,0>(index)]));
//
//	be=abs(cudaDofVector[Mesh.template getCellNextToCell<1,0>(index)]/
//			(cudaDofVector[Mesh.template getCellNextToCell<1,1>(index)]-
//			 cudaDofVector[Mesh.template getCellNextToCell<1,0>(index)]));
//
//	a = al-be;
//	b=1.0;
//	c=-al;
//	s= h/sqrt(a*a+b*b);
//
//
//	cudaDofVector2[index]=fabsMin(-abs(a*0+b*0+c)*s,cudaDofVector2[(index)]);
//	cudaDofVector2[Mesh.template getCellNextToCell<0,1>(index)]=fabsMin(abs(a*0+b*1+c)*s,cudaDofVector2[Mesh.template getCellNextToCell<0,1>(index)]);
//	cudaDofVector2[Mesh.template getCellNextToCell<1,1>(index)]=fabsMin(abs(a*1+b*1+c)*s,cudaDofVector2[Mesh.template getCellNextToCell<1,1>(index)]);
//	cudaDofVector2[Mesh.template getCellNextToCell<1,0>(index)]=fabsMin(-abs(a*1+b*0+c)*s,cudaDofVector2[Mesh.template getCellNextToCell<1,0>(index)]);
//
//}
//
//template< typename MeshReal,
//          typename Device,
//          typename MeshIndex,
//          typename Real,
//          typename Index >
//void tnlNarrowBand< tnlGrid< 3,MeshReal, Device, MeshIndex >, Real, Index > :: setupSquare0101( Index i, Index j)
//{
//	Index index = Mesh.getCellIndex(CoordinatesType(i,j));
//	Real al,be, a,b,c,s;
//	al=abs(cudaDofVector[Mesh.template getCellNextToCell<0,0>(index)]/
//			(cudaDofVector[Mesh.template getCellNextToCell<1,0>(index)]-
//			 cudaDofVector[Mesh.template getCellNextToCell<0,0>(index)]));
//
//	be=abs(cudaDofVector[Mesh.template getCellNextToCell<0,1>(index)]/
//			(cudaDofVector[Mesh.template getCellNextToCell<1,1>(index)]-
//			 cudaDofVector[Mesh.template getCellNextToCell<0,1>(index)]));
//
//	a = al-be;
//	b=1.0;
//	c=-be;
//	s= h/sqrt(a*a+b*b);
//
//
//	cudaDofVector2[index]=fabsMin(-abs(a*0+b*0+c)*s,cudaDofVector2[(index)]);
//	cudaDofVector2[Mesh.template getCellNextToCell<0,1>(index)]=fabsMin(-abs(a*1+b*0+c)*s,cudaDofVector2[Mesh.template getCellNextToCell<0,1>(index)]);
//	cudaDofVector2[Mesh.template getCellNextToCell<1,1>(index)]=fabsMin(abs(a*1+b*1+c)*s,cudaDofVector2[Mesh.template getCellNextToCell<1,1>(index)]);
//	cudaDofVector2[Mesh.template getCellNextToCell<1,0>(index)]=fabsMin(abs(a*0+b*1+c)*s,cudaDofVector2[Mesh.template getCellNextToCell<1,0>(index)]);
//
//}
//
//template< typename MeshReal,
//          typename Device,
//          typename MeshIndex,
//          typename Real,
//          typename Index >
//void tnlNarrowBand< tnlGrid< 3,MeshReal, Device, MeshIndex >, Real, Index > :: setupSquare0110( Index i, Index j)
//{
//	Index index = Mesh.getCellIndex(CoordinatesType(i,j));
//	cudaDofVector2[index]=fabsMin(cudaDofVector[index],cudaDofVector2[(index)]);
//	cudaDofVector2[Mesh.template getCellNextToCell<0,1>(index)]=fabsMin(cudaDofVector[Mesh.template getCellNextToCell<0,1>(index)],cudaDofVector2[Mesh.template getCellNextToCell<0,1>(index)]);
//	cudaDofVector2[Mesh.template getCellNextToCell<1,1>(index)]=fabsMin(cudaDofVector[Mesh.template getCellNextToCell<1,1>(index)],cudaDofVector2[Mesh.template getCellNextToCell<1,1>(index)]);
//	cudaDofVector2[Mesh.template getCellNextToCell<1,0>(index)]=fabsMin(cudaDofVector[Mesh.template getCellNextToCell<1,0>(index)],cudaDofVector2[Mesh.template getCellNextToCell<1,0>(index)]);
//}
#endif




#endif /* TNLNARROWBAND_IMPL_H_ */
