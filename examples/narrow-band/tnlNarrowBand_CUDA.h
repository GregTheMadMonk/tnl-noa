/***************************************************************************
                          tnlNarrowBand_CUDA.h  -  description
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
#ifndef TNLNARROWBAND_H_
#define TNLNARROWBAND_H_

#include <config/tnlParameterContainer.h>
#include <core/vectors/tnlVector.h>
#include <core/vectors/tnlStaticVector.h>
#include <core/tnlHost.h>
#include <mesh/tnlGrid.h>
#include <mesh/grids/tnlGridEntity.h>

#include <functions/tnlMeshFunction.h>
#include <limits.h>
#include <core/tnlDevice.h>
#include <ctime>





template< typename Mesh,
		  typename Real,
		  typename Index >
class tnlNarrowBand
{};




template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
class tnlNarrowBand< tnlGrid< 2,MeshReal, Device, MeshIndex >, Real, Index >
{

public:
	typedef Real RealType;
	typedef Device DeviceType;
	typedef Index IndexType;
	typedef tnlGrid< 2, Real, Device, Index > MeshType;
	typedef tnlVector< RealType, DeviceType, IndexType> DofVectorType;
	typedef typename MeshType::CoordinatesType CoordinatesType;

	tnlNarrowBand();

	__host__ static tnlString getType();
	__host__ bool init( const tnlParameterContainer& parameters );
	__host__ bool run();
#ifdef HAVE_CUDA
   __device__ __host__
#endif
	RealType positivePart(const RealType arg) const;
#ifdef HAVE_CUDA
   __device__ __host__
#endif
	RealType negativePart(const RealType arg) const;

#ifdef HAVE_CUDA
	__device__ bool initGrid();
	__device__ void updateValue(const Index i, const Index j);
	__device__ void updateValue(const Index i, const Index j, double** sharedMem, const int k3);
	__device__ Real fabsMin(const Real x, const Real y);

	tnlNarrowBand< tnlGrid< 2,MeshReal, Device, MeshIndex >, Real, Index >* cudaSolver;
	double* cudaDofVector;
	double* cudaDofVector2;
	int* cudaStatusVector;
	int counter;
	int* reinitialize;
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

	int statusGridSize;
	bool exactInput;

	tnlMeshFunction<MeshType> dofVector;
	DofVectorType data;


	RealType h, tau, finalTime;


};









template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
class tnlNarrowBand< tnlGrid< 3,MeshReal, Device, MeshIndex >, Real, Index >
{

public:
	typedef Real RealType;
	typedef Device DeviceType;
	typedef Index IndexType;
	typedef tnlGrid< 3, Real, Device, Index > MeshType;
	typedef tnlVector< RealType, DeviceType, IndexType> DofVectorType;
	typedef typename MeshType::CoordinatesType CoordinatesType;



	__host__ static tnlString getType();
	__host__ bool init( const tnlParameterContainer& parameters );
	__host__ bool run();

#ifdef HAVE_CUDA
	__device__ bool initGrid(int i, int j, int k);
	__device__ void updateValue(const Index i, const Index j, const Index k);
	__device__ void updateValue(const Index i, const Index j, const Index k, double** sharedMem, const int k3);
	__device__ Real fabsMin(const Real x, const Real y);

	tnlNarrowBand< tnlGrid< 3,MeshReal, Device, MeshIndex >, Real, Index >* cudaSolver;
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
__global__ void runCUDA(tnlNarrowBand< tnlGrid< 2,double, tnlHost, int >, double, int >* solver, int sweep, int i);
//__global__ void runCUDA(tnlNarrowBand< tnlGrid< 3,double, tnlHost, int >, double, int >* solver, int sweep, int i);

__global__ void initCUDA(tnlNarrowBand< tnlGrid< 2,double, tnlHost, int >, double, int >* solver);

__global__ void initSetupGridCUDA(tnlNarrowBand< tnlGrid< 2,double, tnlHost, int >, double, int >* solver);
__global__ void initSetupGrid2CUDA(tnlNarrowBand< tnlGrid< 2,double, tnlHost, int >, double, int >* solver);
__global__ void initSetupGrid1_2CUDA(tnlNarrowBand< tnlGrid< 2,double, tnlHost, int >, double, int >* solver);
__global__ void runNarrowBandCUDA(tnlNarrowBand< tnlGrid< 2,double, tnlHost, int >, double, int >* solver, double tau);
//__global__ void initCUDA(tnlNarrowBand< tnlGrid< 3,double, tnlHost, int >, double, int >* solver);
#endif



#include "tnlNarrowBand2D_CUDA_v5_impl.h"
//											#include "tnlNarrowBand3D_CUDA_impl.h"

#endif /* TNLNARROWBAND_H_ */
