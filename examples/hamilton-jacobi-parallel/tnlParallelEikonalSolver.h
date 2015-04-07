/***************************************************************************
                          tnlParallelEikonalSolver.h  -  description
                             -------------------
    begin                : Nov 28 , 2014
    copyright            : (C) 2014 by Tomas Sobotik
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/

#ifndef TNLPARALLELEIKONALSOLVER_H_
#define TNLPARALLELEIKONALSOLVER_H_

#include <config/tnlParameterContainer.h>
#include <core/vectors/tnlVector.h>
#include <core/vectors/tnlStaticVector.h>
#include <core/tnlHost.h>
#include <mesh/tnlGrid.h>
#include <limits.h>
#include <core/tnlDevice.h>

#ifdef HAVE_CUDA
#include <cuda.h>
#include <core/tnlCuda.h>
#endif


template< typename SchemeHost,
		  typename SchemeDevice,
		  typename Device,
		  typename RealType = double,
          typename IndexType = int >
class tnlParallelEikonalSolver
{};

template< typename SchemeHost, typename SchemeDevice, typename Device>
class tnlParallelEikonalSolver<SchemeHost, SchemeDevice, Device, double, int >
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
	tnlParallelEikonalSolver();
	bool init( const tnlParameterContainer& parameters );
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

	VectorType runSubgrid( int boundaryCondition, VectorType u, int subGridID);


	VectorType u0, work_u;
	IntVectorType subgridValues, boundaryConditions, unusedCell, calculationsCount;
	MeshType mesh, subMesh;
	SchemeHost schemeHost;
	SchemeDevice schemeDevice;
	double delta, tau0, stopTime,cflCondition;
	int gridRows, gridCols, currentStep, n;

	tnlDeviceEnum device;

	tnlParallelEikonalSolver<SchemeHost, SchemeDevice, Device, double, int >* getSelf()
	{
		return this;
	};

#ifdef HAVE_CUDA

	tnlParallelEikonalSolver<SchemeHost, SchemeDevice, Device, double, int >* cudaSolver;

	double* work_u_cuda;

	int* subgridValues_cuda;
	int*boundaryConditions_cuda;
	int* unusedCell_cuda;
	int* calculationsCount_cuda;
	double* tmpw;
	//MeshTypeCUDA mesh_cuda, subMesh_cuda;
	//SchemeDevice scheme_cuda;
	//double delta_cuda, tau0_cuda, stopTime_cuda,cflCondition_cuda;
	//int gridRows_cuda, gridCols_cuda, currentStep_cuda, n_cuda;

	bool* runcuda;
	bool run_host;


	__device__ void getSubgridCUDA( const int i, tnlParallelEikonalSolver<SchemeHost, SchemeDevice, Device, double, int >* caller, double* a);

	__device__ void insertSubgridCUDA( double u, const int i );

	__device__ void runSubgridCUDA( int boundaryCondition, double* u, int subGridID);

	/*__global__ void runCUDA();*/

	//__device__ void synchronizeCUDA();

	__device__ int getOwnerCUDA( int i) const;

	__device__ int getSubgridValueCUDA( int i ) const;

	__device__ void setSubgridValueCUDA( int i, int value );

	__device__ int getBoundaryConditionCUDA( int i ) const;

	__device__ void setBoundaryConditionCUDA( int i, int value );

	//__device__ bool initCUDA( tnlParallelEikonalSolver<SchemeHost, SchemeDevice, Device, double, int >* cudaSolver);

	/*__global__ void initRunCUDA(tnlParallelEikonalSolver<Scheme, double, tnlHost, int >* caller);*/

#endif

};
#ifdef HAVE_CUDA
template <typename SchemeHost, typename SchemeDevice, typename Device>
__global__ void runCUDA(tnlParallelEikonalSolver<SchemeHost, SchemeDevice, Device, double, int >* caller);

template <typename SchemeHost, typename SchemeDevice, typename Device>
__global__ void initRunCUDA(tnlParallelEikonalSolver<SchemeHost, SchemeDevice, Device, double, int >* caller);

template <typename SchemeHost, typename SchemeDevice, typename Device>
__global__ void initCUDA( tnlParallelEikonalSolver<SchemeHost, SchemeDevice, Device, double, int >* cudaSolver, double* ptr, bool * ptr2);

template <typename SchemeHost, typename SchemeDevice, typename Device>
__global__ void synchronizeCUDA(tnlParallelEikonalSolver<SchemeHost, SchemeDevice, Device, double, int >* cudaSolver);
#endif

#include "tnlParallelEikonalSolver_impl.h"

#endif /* TNLPARALLELEIKONALSOLVER_H_ */
