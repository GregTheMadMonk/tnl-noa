/***************************************************************************
                          tnlFastSweepingMap_CUDA.h  -  description
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
#ifndef TNLFASTSWEEPING_H_
#define TNLFASTSWEEPING_H_

#include <TNL/Config/ParameterContainer.h>
#include <TNL/Containers/Vector.h>
#include <TNL/Containers/StaticVector.h>
#include <TNL/Devices/Host.h>
#include <mesh/tnlGrid.h>
#include <mesh/grids/tnlGridEntity.h>

#include <functions/tnlMeshFunction.h>
#include <limits.h>
#include <core/tnlDevice.h>
#include <ctime>





template< typename Mesh,
		  typename Real,
		  typename Index >
class tnlFastSweepingMap
{};




template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
class tnlFastSweepingMap< tnlGrid< 2,MeshReal, Device, MeshIndex >, Real, Index >
{

public:
	typedef Real RealType;
	typedef Device DeviceType;
	typedef Index IndexType;
	typedef tnlGrid< 2, Real, Device, Index > MeshType;
	typedef TNL::Containers::Vector< RealType, DeviceType, IndexType> DofVectorType;
	typedef typename MeshType::CoordinatesType CoordinatesType;

	tnlFastSweepingMap();

	__host__ static String getType();
	__host__ bool init( const Config::ParameterContainer& parameters );
	__host__ bool run();

#ifdef HAVE_CUDA
	__device__ bool initGrid();
	__device__ void updateValue(const Index i, const Index j, Index* something_changed);
	__device__ void updateValue(const Index i, const Index j, double** sharedMem, const int k3);
	__device__ Real fabsMin(const Real x, const Real y);

	tnlFastSweepingMap< tnlGrid< 2,MeshReal, Device, MeshIndex >, Real, Index >* cudaSolver;
	double* cudaDofVector;
	double* cudaDofVector2;
	double* map_cuda;
	int counter;
	int* changed;
	__device__ void setupSquare1000(Index i, Index j);
	__device__ void setupSquare1100(Index i, Index j);
	__device__ void setupSquare1010(Index i, Index j);
	__device__ void setupSquare1001(Index i, Index j);
	__device__ void setupSquare1110(Index i, Index j);
	__device__ void setupSquare1101(Index i, Index j);
	__device__ void setupSquare1011(Index i, Index j);
	__device__ void setupSquare1111(Index i, Index j);
	__device__ void setupSquare0000(Index i, Index j);
	__device__ void setupSquare0100(Index i, Index j);
	__device__ void setupSquare0010(Index i, Index j);
	__device__ void setupSquare0001(Index i, Index j);
	__device__ void setupSquare0110(Index i, Index j);
	__device__ void setupSquare0101(Index i, Index j);
	__device__ void setupSquare0011(Index i, Index j);
	__device__ void setupSquare0111(Index i, Index j);
#endif

	MeshType Mesh;

protected:



	bool exactInput;

	tnlMeshFunction<MeshType> dofVector;
	DofVectorType data, map;


	RealType h;


};









template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
class tnlFastSweepingMap< tnlGrid< 3,MeshReal, Device, MeshIndex >, Real, Index >
{

public:
	typedef Real RealType;
	typedef Device DeviceType;
	typedef Index IndexType;
	typedef tnlGrid< 3, Real, Device, Index > MeshType;
	typedef TNL::Containers::Vector< RealType, DeviceType, IndexType> DofVectorType;
	typedef typename MeshType::CoordinatesType CoordinatesType;



	__host__ static String getType();
	__host__ bool init( const Config::ParameterContainer& parameters );
	__host__ bool run();

#ifdef HAVE_CUDA
	__device__ bool initGrid(int i, int j, int k);
	__device__ void updateValue(const Index i, const Index j, const Index k);
	__device__ void updateValue(const Index i, const Index j, const Index k, double** sharedMem, const int k3);
	__device__ Real fabsMin(const Real x, const Real y);

	tnlFastSweepingMap< tnlGrid< 3,MeshReal, Device, MeshIndex >, Real, Index >* cudaSolver;
	double* cudaDofVector;
	double* cudaDofVector2;
	int counter;
#endif

	MeshType Mesh;

protected:



	bool exactInput;

	tnlMeshFunction<MeshType> dofVector;
	DofVectorType data;

	RealType h;


};







#ifdef HAVE_CUDA
//template<int sweep_t>
__global__ void runCUDA(tnlFastSweepingMap< tnlGrid< 2,double, TNL::Devices::Host, int >, double, int >* solver, int sweep, int i, int* changed);
//__global__ void runCUDA(tnlFastSweepingMap< tnlGrid< 3,double, TNL::Devices::Host, int >, double, int >* solver, int sweep, int i);

__global__ void initCUDA(tnlFastSweepingMap< tnlGrid< 2,double, TNL::Devices::Host, int >, double, int >* solver);
//__global__ void initCUDA(tnlFastSweepingMap< tnlGrid< 3,double, TNL::Devices::Host, int >, double, int >* solver);
#endif

/*various implementtions.... choose one*/
//#include "tnlFastSweepingMap2D_CUDA_impl.h"
//#include "tnlFastSweepingMap2D_CUDA_v2_impl.h"
//#include "tnlFastSweepingMap2D_CUDA_v3_impl.h"
#include "tnlFastSweepingMap2D_CUDA_v4_impl.h"
//#include "tnlFastSweepingMap2D_CUDA_v5_impl.h"


//															#include "tnlFastSweepingMap3D_CUDA_impl.h"

#endif /* TNLFASTSWEEPING_H_ */
