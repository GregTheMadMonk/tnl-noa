/***************************************************************************
                          tnlFastSweeping.h  -  description
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

#include <config/tnlParameterContainer.h>
#include <core/vectors/tnlVector.h>
#include <core/vectors/tnlStaticVector.h>
#include <core/tnlHost.h>
#include <mesh/tnlGrid.h>
#include <limits.h>
#include <core/tnlDevice.h>
#include <ctime>
#ifdef HAVE_OPENMP
#include <omp.h>
#endif




template< typename Mesh,
		  typename Real,
		  typename Index >
class tnlFastSweeping
{};




template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
class tnlFastSweeping< tnlGrid< 2,MeshReal, Device, MeshIndex >, Real, Index >
{

public:
	typedef Real RealType;
	typedef Device DeviceType;
	typedef Index IndexType;
	typedef tnlGrid< 2, Real, Device, Index > MeshType;
	typedef tnlVector< RealType, DeviceType, IndexType> DofVectorType;
	typedef typename MeshType::CoordinatesType CoordinatesType;



	static tnlString getType();
	bool init( const tnlParameterContainer& parameters );

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

	DofVectorType dofVector, dofVector2;

	RealType h;

#ifdef HAVE_OPENMP
//	omp_lock_t* gridLock;
#endif


};

	//for single core version use this implementation:
#include "tnlFastSweeping2D_impl.h"
	//for parallel version use this one instead:
// #include "tnlFastSweeping2D_openMP_impl.h"

#endif /* TNLFASTSWEEPING_H_ */
