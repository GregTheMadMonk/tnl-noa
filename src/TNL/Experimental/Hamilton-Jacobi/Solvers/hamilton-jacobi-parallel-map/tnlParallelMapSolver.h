/***************************************************************************
                          tnlParallelMapSolver.h  -  description
                             -------------------
    begin                : Mar 22 , 2016
    copyright            : (C) 2016 by Tomas Sobotik
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/

#ifndef TNLPARALLELMAPSOLVER_H_
#define TNLPARALLELMAPSOLVER_H_

#include <TNL/Config/ParameterContainer.h>
#include <core/vectors/tnlVector.h>
#include <TNL/Containers/StaticVector.h>
#include <functions/tnlMeshFunction.h>
#include <core/tnlHost.h>
#include <mesh/tnlGrid.h>
#include <mesh/grids/tnlGridEntity.h>
#include <limits.h>
#include <core/tnlDevice.h>


#include <ctime>

#ifdef HAVE_CUDA
#include <core/tnlCuda.h>
#endif


template< int Dimension,
		  typename SchemeHost,
		  typename SchemeDevice,
		  typename Device,
		  typename RealType = double,
          typename IndexType = int >
class tnlParallelMapSolver
{};

template<typename SchemeHost, typename SchemeDevice, typename Device>
class tnlParallelMapSolver<2, SchemeHost, SchemeDevice, Device, double, int >
{
public:

	typedef SchemeDevice SchemeTypeDevice;
	typedef SchemeHost SchemeTypeHost;
	typedef Device DeviceType;
	typedef tnlVector< double, tnlHost, int > VectorType;
	typedef tnlVector< int, tnlHost, int > IntVectorType;
	typedef tnlGrid< 2, double, tnlHost, int > MeshType;
#ifdef HAVE_CUDA
	typedef tnlVector< double, tnlHost, int > VectorTypeCUDA;
	typedef tnlVector< int, tnlHost, int > IntVectorTypeCUDA;
	typedef tnlGrid< 2, double, tnlHost, int > MeshTypeCUDA;
#endif
	tnlParallelMapSolver();
	bool init( const Config::ParameterContainer& parameters );
	void run();

	void test();

/*private:*/


	void synchronize();

	int getOwner( int i) const;

	int getSubgridValue( int i ) const;

	void setSubgridValue( int i, int value );

	int getBoundaryCondition( int i ) const;

	void setBoundaryCondition( int i, int value );

	void stretchGrid();

	void contractGrid();

	VectorType getSubgrid( const int i ) const;

	void insertSubgrid( VectorType u, const int i );

	VectorType runSubgrid( int boundaryCondition, VectorType u, int subGridID,VectorType map);


	tnlMeshFunction<MeshType> u0;
	VectorType work_u, map_stretched, map;
	IntVectorType subgridValues, boundaryConditions, unusedCell, calculationsCount;
	MeshType mesh, subMesh;

//	tnlGridEntity< MeshType, 2, tnlGridEntityNoStencilStorage > Entity;

	SchemeHost schemeHost;
	SchemeDevice schemeDevice;
	double delta, tau0, stopTime,cflCondition;
	int gridRows, gridCols, gridLevels, currentStep, n;

	std::clock_t start;
	double time_diff;


	tnlDeviceEnum device;

	tnlParallelMapSolver<2, SchemeHost, SchemeDevice, Device, double, int >* getSelf()
	{
		return this;
	};

#ifdef HAVE_CUDA

	tnlParallelMapSolver<2, SchemeHost, SchemeDevice, Device, double, int >* cudaSolver;

	double* work_u_cuda;
	double* map_stretched_cuda;

	int* subgridValues_cuda;
	int* boundaryConditions_cuda;
	int* unusedCell_cuda;
	int* calculationsCount_cuda;
	double* tmpw;
	double* tmp_map;


	int* runcuda;
	int run_host;


	__device__ void getSubgridCUDA2D( const int i, tnlParallelMapSolver<2, SchemeHost, SchemeDevice, Device, double, int >* caller, double* a);

	__device__ void updateSubgridCUDA2D( const int i, tnlParallelMapSolver<2, SchemeHost, SchemeDevice, Device, double, int >* caller, double* a);

	__device__ void insertSubgridCUDA2D( double u, const int i );

	__device__ void runSubgridCUDA2D( int boundaryCondition, double* u, int subGridID);

	__device__ int getOwnerCUDA2D( int i) const;

	__device__ int getSubgridValueCUDA2D( int i ) const;

	__device__ void setSubgridValueCUDA2D( int i, int value );

	__device__ int getBoundaryConditionCUDA2D( int i ) const;

	__device__ void setBoundaryConditionCUDA2D( int i, int value );

#endif

};














#ifdef HAVE_CUDA
template <typename SchemeHost, typename SchemeDevice, typename Device>
__global__ void runCUDA2D(tnlParallelMapSolver<2, SchemeHost, SchemeDevice, Device, double, int >* caller);

template <typename SchemeHost, typename SchemeDevice, typename Device>
__global__ void initRunCUDA2D(tnlParallelMapSolver<2, SchemeHost, SchemeDevice, Device, double, int >* caller);

template <typename SchemeHost, typename SchemeDevice, typename Device>
__global__ void initCUDA2D( tnlParallelMapSolver<2, SchemeHost, SchemeDevice, Device, double, int >* cudaSolver, double* ptr, int * ptr2, int* ptr3, double* tmp_map_ptr);

template <typename SchemeHost, typename SchemeDevice, typename Device>
__global__ void synchronizeCUDA2D(tnlParallelMapSolver<2, SchemeHost, SchemeDevice, Device, double, int >* cudaSolver);

template <typename SchemeHost, typename SchemeDevice, typename Device>
__global__ void synchronize2CUDA2D(tnlParallelMapSolver<2, SchemeHost, SchemeDevice, Device, double, int >* cudaSolver);



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

#endif

#include "tnlParallelMapSolver2D_impl.h"
#endif /* TNLPARALLELMAPSOLVER_H_ */
