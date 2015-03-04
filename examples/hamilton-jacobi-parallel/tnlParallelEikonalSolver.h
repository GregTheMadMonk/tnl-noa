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


template< typename Scheme,
		  typename RealType = double,
		  typename Device = tnlHost,
          typename IndexType = int >
class tnlParallelEikonalSolver
{};

template< typename Scheme>
class tnlParallelEikonalSolver<Scheme, double, tnlHost, int >
{
public:

	typedef tnlVector< double, tnlHost, int > VectorType;
	typedef tnlVector< int, tnlHost, int > IntVectorType;
	typedef tnlGrid< 2, double, tnlHost, int > MeshType;

	tnlParallelEikonalSolver();
	bool init( const tnlParameterContainer& parameters );
	void run();

	void test();

private:


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
	IntVectorType subgridValues, boundaryConditions, unusedCell;
	MeshType mesh, subMesh;
	Scheme scheme;
	double delta, tau0, stopTime,cflCondition;
	int gridRows, gridCols, currentStep, n;

};

#include "tnlParallelEikonalSolver_impl.h"

#endif /* TNLPARALLELEIKONALSOLVER_H_ */
