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
	void updateValue(const Index i, const Index j);
	Real fabsMin(const Real x, const Real y);


protected:

	MeshType Mesh;

	DofVectorType dofVector;

	RealType h;


};


#include "tnlFastSweeping2D_impl.h"

#endif /* TNLFASTSWEEPING_H_ */
