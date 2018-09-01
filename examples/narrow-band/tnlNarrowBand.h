/***************************************************************************
                          tnlNarrowBand.h  -  description
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

#include <TNL/Config/ParameterContainer.h>
#include <TNL/Containers/Vector.h>
#include <TNL/Containers/StaticVector.h>
#include <functions/tnlMeshFunction.h>
#include <TNL/Devices/Host.h>
#include <mesh/tnlGrid.h>
#include <mesh/grids/tnlGridEntity.h>
#include <limits.h>
#include <core/tnlDevice.h>
#include <ctime>
#ifdef HAVE_OPENMP
#include <omp.h>
#endif




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
	typedef TNL::Containers::Vector< RealType, DeviceType, IndexType> DofVectorType;
	typedef typename MeshType::CoordinatesType CoordinatesType;


	tnlNarrowBand();

	static String getType();
	bool init( const Config::ParameterContainer& parameters );

	bool initGrid();
	bool run();

	//for single core version use this implementation:
	void updateValue(const Index i, const Index j);
	//for parallel version use this one instead:
//	void updateValue(const Index i, const Index j, DofVectorType* grid);


	void setupSquare1000(Index i, Index j);
	void setupSquare1100(Index i, Index j);
	void setupSquare1010(Index i, Index j);
	void setupSquare1001(Index i, Index j);
	void setupSquare1110(Index i, Index j);
	void setupSquare1101(Index i, Index j);
	void setupSquare1011(Index i, Index j);
	void setupSquare1111(Index i, Index j);
	void setupSquare0000(Index i, Index j);
	void setupSquare0100(Index i, Index j);
	void setupSquare0010(Index i, Index j);
	void setupSquare0001(Index i, Index j);
	void setupSquare0110(Index i, Index j);
	void setupSquare0101(Index i, Index j);
	void setupSquare0011(Index i, Index j);
	void setupSquare0111(Index i, Index j);

	Real fabsMin(const Real x, const Real y);


protected:

	MeshType Mesh;

	bool exactInput;

	tnlMeshFunction<MeshType> dofVector, dofVector2;
	DofVectorType data;

	RealType h;

	tnlGridEntity< MeshType, 2, tnlGridEntityNoStencilStorage > Entity;


#ifdef HAVE_OPENMP
//	omp_lock_t* gridLock;
#endif


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
	typedef TNL::Containers::Vector< RealType, DeviceType, IndexType> DofVectorType;
	typedef typename MeshType::CoordinatesType CoordinatesType;

	tnlNarrowBand();

	static String getType();
	bool init( const Config::ParameterContainer& parameters );

	bool initGrid();
	bool run();

	//for single core version use this implementation:
	void updateValue(const Index i, const Index j, const Index k);
	//for parallel version use this one instead:
//	void updateValue(const Index i, const Index j, DofVectorType* grid);

	Real fabsMin(const Real x, const Real y);


protected:

	MeshType Mesh;

	bool exactInput;


	tnlMeshFunction<MeshType> dofVector, dofVector2;
	DofVectorType data;

	RealType h;

	tnlGridEntity< MeshType, 3, tnlGridEntityNoStencilStorage > Entity;

#ifdef HAVE_OPENMP
//	omp_lock_t* gridLock;
#endif


};


	//for single core version use this implementation:
#include "tnlNarrowBand2D_impl.h"
	//for parallel version use this one instead:
// #include "tnlNarrowBand2D_openMP_impl.h"

#include "tnlNarrowBand3D_impl.h"

#endif /* TNLNARROWBAND_H_ */
