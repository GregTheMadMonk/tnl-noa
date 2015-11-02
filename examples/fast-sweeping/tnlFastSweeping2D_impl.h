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
tnlString tnlFastSweeping< tnlGrid< 2,MeshReal, Device, MeshIndex >, Real, Index > :: getType()
{
	   return tnlString( "tnlFastSweeping< " ) +
	          MeshType::getType() + ", " +
	          ::getType< Real >() + ", " +
	          ::getType< Index >() + " >";
}




template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
bool tnlFastSweeping< tnlGrid< 2,MeshReal, Device, MeshIndex >, Real, Index > :: init( const tnlParameterContainer& parameters )
{
	const tnlString& meshFile = parameters.getParameter< tnlString >( "mesh" );

	if( ! Mesh.load( meshFile ) )
	{
		   cerr << "I am not able to load the mesh from the file " << meshFile << "." << endl;
		   return false;
	}


	const tnlString& initialCondition = parameters.getParameter <tnlString>("initial-condition");
	if( ! dofVector.load( initialCondition ) )
	{
		   cerr << "I am not able to load the initial condition from the file " << meshFile << "." << endl;
		   return false;
	}

	h = Mesh.getHx();

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

	for(Index i = 0; i < Mesh.getDimensions().x(); i++)
	{
		for(Index j = 0; j < Mesh.getDimensions().y(); j++)
		{
			updateValue(i,j);
		}
	}

/*---------------------------------------------------------------------------------------------------------------------------*/

	for(Index i = Mesh.getDimensions().x() - 1; i > -1; i--)
	{
		for(Index j = 0; j < Mesh.getDimensions().y(); j++)
		{
			updateValue(i,j);
		}
	}

/*---------------------------------------------------------------------------------------------------------------------------*/

	for(Index i = Mesh.getDimensions().x() - 1; i > -1; i--)
	{
		for(Index j = Mesh.getDimensions().y() - 1; j > -1; j--)
		{
			updateValue(i,j);
		}
	}

/*---------------------------------------------------------------------------------------------------------------------------*/
	for(Index i = 0; i < Mesh.getDimensions().x(); i++)
	{
		for(Index j = Mesh.getDimensions().y() - 1; j > -1; j--)
		{
			updateValue(i,j);
		}
	}

/*---------------------------------------------------------------------------------------------------------------------------*/


	dofVector.save("u-00001.tnl");

	return true;
}


template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
void tnlFastSweeping< tnlGrid< 2,MeshReal, Device, MeshIndex >, Real, Index > :: updateValue( Index i, Index j)
{
	Index index = Mesh.getCellIndex(CoordinatesType(i,j));
	Real value = dofVector[index];
	Real a,b, tmp;

	if( i == 0 )
		a = dofVector[Mesh.template getCellNextToCell<1,0>(index)];
	else if( i == Mesh.getDimensions().x() - 1 )
		a = dofVector[Mesh.template getCellNextToCell<-1,0>(index)];
	else
	{
		a = fabsMin( dofVector[Mesh.template getCellNextToCell<-1,0>(index)],
				 dofVector[Mesh.template getCellNextToCell<1,0>(index)] );
	}

	if( j == 0 )
		b = dofVector[Mesh.template getCellNextToCell<0,1>(index)];
	else if( j == Mesh.getDimensions().y() - 1 )
		b = dofVector[Mesh.template getCellNextToCell<0,-1>(index)];
	else
	{
		b = fabsMin( dofVector[Mesh.template getCellNextToCell<0,-1>(index)],
				 dofVector[Mesh.template getCellNextToCell<0,1>(index)] );
	}


	if(fabs(a-b) >= h)
		tmp = fabsMin(a,b) + Sign(value)*h;
	else
		tmp = 0.5 * (a + b + Sign(value)*sqrt(2.0 * h * h - (a - b) * (a - b) ) );


	dofVector[index]  = fabsMin(value, tmp);
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
