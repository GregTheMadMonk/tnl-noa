/***************************************************************************
                          tnlFastSweeping_impl.h  -  description
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

#include "tnlFastSweeping.h"

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
String tnlFastSweeping< tnlGrid< 2,MeshReal, Device, MeshIndex >, Real, Index > :: getType()
{
	   return String( "tnlFastSweeping< " ) +
	          MeshType::getType() + ", " +
	          ::getType< Real >() + ", " +
	          ::getType< Index >() + " >";
}




template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
bool tnlFastSweeping< tnlGrid< 2,MeshReal, Device, MeshIndex >, Real, Index > :: init( const Config::ParameterContainer& parameters )
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

	h = Mesh.getSpaceSteps().x();


	const String& exact_input = parameters.getParameter< String >( "exact-input" );

	if(exact_input == "no")
		exactInput=false;
	else
		exactInput=true;

#ifdef HAVE_OPENMP
//	gridLock = (omp_lock_t*) malloc(sizeof(omp_lock_t)*Mesh.getDimensions().x()*Mesh.getDimensions().y());
//
//	for(int i = 0; i < Mesh.getDimensions().x()*Mesh.getDimensions().y(); i++)
//			omp_init_lock(&gridLock[i]);
#endif

	return initGrid();
}


template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
bool tnlFastSweeping< tnlGrid< 2,MeshReal, Device, MeshIndex >, Real, Index > :: initGrid()
{

	Real tmp = 0.0;

	if(!exactInput)
	{
		for(Index i = 0; i < Mesh.getDimensions().x()*Mesh.getDimensions().y(); i++)
				dofVector[i]=0.5*h*Sign(dofVector[i]);
	}


	for(Index i = 1; i < Mesh.getDimensions().x()-1; i++)
	{
		for(Index j = 1; j < Mesh.getDimensions().y()-1; j++)
		{
			 tmp = Sign(dofVector[Mesh.getCellIndex(CoordinatesType(i,j))]);


			if(tmp == 0.0)
			{}
			else if(dofVector[Mesh.getCellIndex(CoordinatesType(i+1,j))]*tmp < 0.0 ||
					dofVector[Mesh.getCellIndex(CoordinatesType(i-1,j))]*tmp < 0.0 ||
					dofVector[Mesh.getCellIndex(CoordinatesType(i,j+1))]*tmp < 0.0 ||
					dofVector[Mesh.getCellIndex(CoordinatesType(i,j-1))]*tmp < 0.0 )
			{}
			else
				dofVector[Mesh.getCellIndex(CoordinatesType(i,j))] = tmp*INT_MAX;
		}
	}



	for(int i = 1; i < Mesh.getDimensions().x()-1; i++)
	{
		Index j = 0;
		tmp = Sign(dofVector[Mesh.getCellIndex(CoordinatesType(i,j))]);


		if(tmp == 0.0)
		{}
		else if(dofVector[Mesh.getCellIndex(CoordinatesType(i+1,j))]*tmp < 0.0 ||
				dofVector[Mesh.getCellIndex(CoordinatesType(i-1,j))]*tmp < 0.0 ||
				dofVector[Mesh.getCellIndex(CoordinatesType(i,j+1))]*tmp < 0.0 )
		{}
		else
			dofVector[Mesh.getCellIndex(CoordinatesType(i,j))] = tmp*INT_MAX;
	}

	for(int i = 1; i < Mesh.getDimensions().x()-1; i++)
	{
		Index j = Mesh.getDimensions().y() - 1;
		tmp = Sign(dofVector[Mesh.getCellIndex(CoordinatesType(i,j))]);


		if(tmp == 0.0)
		{}
		else if(dofVector[Mesh.getCellIndex(CoordinatesType(i+1,j))]*tmp < 0.0 ||
				dofVector[Mesh.getCellIndex(CoordinatesType(i-1,j))]*tmp < 0.0 ||
				dofVector[Mesh.getCellIndex(CoordinatesType(i,j-1))]*tmp < 0.0 )
		{}
		else
			dofVector[Mesh.getCellIndex(CoordinatesType(i,j))] = tmp*INT_MAX;
	}

	for(int j = 1; j < Mesh.getDimensions().y()-1; j++)
	{
		Index i = 0;
		tmp = Sign(dofVector[Mesh.getCellIndex(CoordinatesType(i,j))]);


		if(tmp == 0.0)
		{}
		else if(dofVector[Mesh.getCellIndex(CoordinatesType(i+1,j))]*tmp < 0.0 ||
				dofVector[Mesh.getCellIndex(CoordinatesType(i,j+1))]*tmp < 0.0 ||
				dofVector[Mesh.getCellIndex(CoordinatesType(i,j-1))]*tmp < 0.0 )
		{}
		else
			dofVector[Mesh.getCellIndex(CoordinatesType(i,j))] = tmp*INT_MAX;
	}

	for(int j = 1; j < Mesh.getDimensions().y()-1; j++)
	{
		Index i = Mesh.getDimensions().x() - 1;
		tmp = Sign(dofVector[Mesh.getCellIndex(CoordinatesType(i,j))]);


		if(tmp == 0.0)
		{}
		else if(dofVector[Mesh.getCellIndex(CoordinatesType(i-1,j))]*tmp < 0.0 ||
				dofVector[Mesh.getCellIndex(CoordinatesType(i,j+1))]*tmp < 0.0 ||
				dofVector[Mesh.getCellIndex(CoordinatesType(i,j-1))]*tmp < 0.0 )
		{}
		else
			dofVector[Mesh.getCellIndex(CoordinatesType(i,j))] = tmp*INT_MAX;
	}


	Index i = Mesh.getDimensions().x() - 1;
	Index j = Mesh.getDimensions().y() - 1;

	tmp = Sign(dofVector[Mesh.getCellIndex(CoordinatesType(i,j))]);
	if(dofVector[Mesh.getCellIndex(CoordinatesType(i-1,j))]*tmp > 0.0 &&
			dofVector[Mesh.getCellIndex(CoordinatesType(i,j-1))]*tmp > 0.0)

		dofVector[Mesh.getCellIndex(CoordinatesType(i,j))] = tmp*INT_MAX;



	j = 0;
	tmp = Sign(dofVector[Mesh.getCellIndex(CoordinatesType(i,j))]);
	if(dofVector[Mesh.getCellIndex(CoordinatesType(i-1,j))]*tmp > 0.0 &&
			dofVector[Mesh.getCellIndex(CoordinatesType(i,j+1))]*tmp > 0.0)

		dofVector[Mesh.getCellIndex(CoordinatesType(i,j))] = tmp*INT_MAX;



	i = 0;
	j = Mesh.getDimensions().y() -1;
	tmp = Sign(dofVector[Mesh.getCellIndex(CoordinatesType(i,j))]);
	if(dofVector[Mesh.getCellIndex(CoordinatesType(i+1,j))]*tmp > 0.0 &&
			dofVector[Mesh.getCellIndex(CoordinatesType(i,j-1))]*tmp > 0.0)

		dofVector[Mesh.getCellIndex(CoordinatesType(i,j))] = tmp*INT_MAX;



	j = 0;
	tmp = Sign(dofVector[Mesh.getCellIndex(CoordinatesType(i,j))]);
	if(dofVector[Mesh.getCellIndex(CoordinatesType(i+1,j))]*tmp > 0.0 &&
			dofVector[Mesh.getCellIndex(CoordinatesType(i,j+1))]*tmp > 0.0)

		dofVector[Mesh.getCellIndex(CoordinatesType(i,j))] = tmp*INT_MAX;


	dofVector.save("u-00000.tnl");

	return true;
}



template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
bool tnlFastSweeping< tnlGrid< 2,MeshReal, Device, MeshIndex >, Real, Index > :: run()
{

	DofVectorType d2,d3,d4;
	d2.setLike(dofVector);
	d2=dofVector;
	d3.setLike(dofVector);
	d3=dofVector;
	d4.setLike(dofVector);
	d4=dofVector;


#ifdef HAVE_OPENMP
#pragma omp parallel sections num_threads(4)
	{
	{
#endif

	for(Index i = 0; i < Mesh.getDimensions().x(); i++)
	{
		for(Index j = 0; j < Mesh.getDimensions().y(); j++)
		{
			updateValue(i,j,&dofVector);
		}
	}

/*---------------------------------------------------------------------------------------------------------------------------*/
#ifdef HAVE_OPENMP
	}
#pragma omp section
	{
#endif
	for(Index i = Mesh.getDimensions().x() - 1; i > -1; i--)
	{
		for(Index j = 0; j < Mesh.getDimensions().y(); j++)
		{
			updateValue(i,j,&d2);
		}
	}

/*---------------------------------------------------------------------------------------------------------------------------*/
#ifdef HAVE_OPENMP
	}
#pragma omp section
	{
#endif
	for(Index i = Mesh.getDimensions().x() - 1; i > -1; i--)
	{
		for(Index j = Mesh.getDimensions().y() - 1; j > -1; j--)
		{
			updateValue(i,j, &d3);
		}
	}

/*---------------------------------------------------------------------------------------------------------------------------*/
#ifdef HAVE_OPENMP
	}
#pragma omp section
	{
#endif
	for(Index i = 0; i < Mesh.getDimensions().x(); i++)
	{
		for(Index j = Mesh.getDimensions().y() - 1; j > -1; j--)
		{
			updateValue(i,j, &d4);
		}
	}

/*---------------------------------------------------------------------------------------------------------------------------*/
#ifdef HAVE_OPENMP
	}
	}
#endif


#ifdef HAVE_OPENMP
#pragma omp parallel for num_threads(4) schedule(dynamic)
#endif
	for(Index i = 0; i < Mesh.getDimensions().x(); i++)
	{
		for(Index j = Mesh.getDimensions().y() - 1; j > -1; j--)
		{
			int index = Mesh.getCellIndex(CoordinatesType(i,j));
			dofVector[index] = fabsMin(dofVector[index], d2[index]);
			dofVector[index] = fabsMin(dofVector[index], d3[index]);
			dofVector[index] = fabsMin(dofVector[index], d4[index]);
		}
	}

	dofVector.save("u-00001.tnl");

	return true;
}


template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
void tnlFastSweeping< tnlGrid< 2,MeshReal, Device, MeshIndex >, Real, Index > :: updateValue( Index i, Index j, DofVectorType* grid)
{
	Index index = Mesh.getCellIndex(CoordinatesType(i,j));
	Real value = (*grid)[index];
	Real a,b, tmp;

	if( i == 0 )
		a = (*grid)[Mesh.template getCellNextToCell<1,0>(index)];
	else if( i == Mesh.getDimensions().x() - 1 )
		a = (*grid)[Mesh.template getCellNextToCell<-1,0>(index)];
	else
	{
		a = fabsMin( (*grid)[Mesh.template getCellNextToCell<-1,0>(index)],
				 (*grid)[Mesh.template getCellNextToCell<1,0>(index)] );
	}

	if( j == 0 )
		b = (*grid)[Mesh.template getCellNextToCell<0,1>(index)];
	else if( j == Mesh.getDimensions().y() - 1 )
		b = (*grid)[Mesh.template getCellNextToCell<0,-1>(index)];
	else
	{
		b = fabsMin( (*grid)[Mesh.template getCellNextToCell<0,-1>(index)],
				 (*grid)[Mesh.template getCellNextToCell<0,1>(index)] );
	}


	if(fabs(a-b) >= h)
		tmp = fabsMin(a,b) + Sign(value)*h;
	else
		tmp = 0.5 * (a + b + Sign(value)*sqrt(2.0 * h * h - (a - b) * (a - b) ) );

#ifdef HAVE_OPENMP
//	omp_set_lock(&gridLock[index]);
#endif
	(*grid)[index]  = fabsMin(value, tmp);
#ifdef HAVE_OPENMP
//	omp_unset_lock(&gridLock[index]);
#endif
}


template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
Real tnlFastSweeping< tnlGrid< 2,MeshReal, Device, MeshIndex >, Real, Index > :: fabsMin( Real x, Real y)
{
	Real fx = fabs(x);
	Real fy = fabs(y);

	Real tmpMin = Min(fx,fy);

	if(tmpMin == fx)
		return x;
	else
		return y;


}




#endif /* TNLFASTSWEEPING_IMPL_H_ */
