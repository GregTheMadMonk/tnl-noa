/***************************************************************************
                          tnlFastSweeping2D_impl.h  -  description
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
#ifndef TNLFASTSWEEPING3D_IMPL_H_
#define TNLFASTSWEEPING3D_IMPL_H_

#include "tnlFastSweeping.h"

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
String tnlFastSweeping< tnlGrid< 3,MeshReal, Device, MeshIndex >, Real, Index > :: getType()
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
tnlFastSweeping< tnlGrid< 3,MeshReal, Device, MeshIndex >, Real, Index > :: tnlFastSweeping()
:Entity(Mesh),
 dofVector(Mesh),
 dofVector2(Mesh)
{
}

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
bool tnlFastSweeping< tnlGrid< 3,MeshReal, Device, MeshIndex >, Real, Index > :: init( const Config::ParameterContainer& parameters )
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
	dofVector2.load(initialCondition);

	h = Mesh.template getSpaceStepsProducts< 1, 0, 0 >();
	Entity.refresh();

	const String& exact_input = parameters.getParameter< String >( "exact-input" );

	if(exact_input == "no")
		exactInput=false;
	else
		exactInput=true;
//	cout << "bla "<<endl;
	return initGrid();
}


template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
bool tnlFastSweeping< tnlGrid< 3,MeshReal, Device, MeshIndex >, Real, Index > :: initGrid()
{
	for(int i=0; i< Mesh.getDimensions().x()*Mesh.getDimensions().y()*Mesh.getDimensions().z();i++)
	{

		if (abs(dofVector[i]) < 1.8*h)
			dofVector2[i]=dofVector[i];
		else
			dofVector2[i]=INT_MAX*sign(dofVector[i]);
	}

	return true;
}



template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
bool tnlFastSweeping< tnlGrid< 3,MeshReal, Device, MeshIndex >, Real, Index > :: run()
{

	for(Index k = 0; k < Mesh.getDimensions().z(); k++)
	{
		for(Index i = 0; i < Mesh.getDimensions().x(); i++)
		{
			for(Index j = 0; j < Mesh.getDimensions().y(); j++)
			{
				updateValue(i,j,k);
			}
		}
	}

/*---------------------------------------------------------------------------------------------------------------------------*/

	for(Index k = 0; k < Mesh.getDimensions().z(); k++)
	{
		for(Index i = Mesh.getDimensions().x() - 1; i > -1; i--)
		{
			for(Index j = 0; j < Mesh.getDimensions().y(); j++)
			{
				updateValue(i,j,k);
			}
		}
	}

/*---------------------------------------------------------------------------------------------------------------------------*/

	for(Index k = 0; k < Mesh.getDimensions().z(); k++)
	{
		for(Index i = Mesh.getDimensions().x() - 1; i > -1; i--)
		{
			for(Index j = Mesh.getDimensions().y() - 1; j > -1; j--)
			{
				updateValue(i,j,k);
			}
		}
	}

/*---------------------------------------------------------------------------------------------------------------------------*/
	for(Index k = 0; k < Mesh.getDimensions().z(); k++)
	{
		for(Index i = 0; i < Mesh.getDimensions().x(); i++)
		{
			for(Index j = Mesh.getDimensions().y() - 1; j > -1; j--)
			{
				updateValue(i,j,k);
			}
		}
	}

/*---------------------------------------------------------------------------------------------------------------------------*/








	for(Index k = Mesh.getDimensions().z() -1; k > -1; k--)
	{
		for(Index i = 0; i < Mesh.getDimensions().x(); i++)
		{
			for(Index j = 0; j < Mesh.getDimensions().y(); j++)
			{
				updateValue(i,j,k);
			}
		}
	}

/*---------------------------------------------------------------------------------------------------------------------------*/

	for(Index k = Mesh.getDimensions().z() -1; k > -1; k--)
	{
		for(Index i = Mesh.getDimensions().x() - 1; i > -1; i--)
		{
			for(Index j = 0; j < Mesh.getDimensions().y(); j++)
			{
				updateValue(i,j,k);
			}
		}
	}

/*---------------------------------------------------------------------------------------------------------------------------*/

	for(Index k = Mesh.getDimensions().z() -1; k > -1; k--)
	{
		for(Index i = Mesh.getDimensions().x() - 1; i > -1; i--)
		{
			for(Index j = Mesh.getDimensions().y() - 1; j > -1; j--)
			{
				updateValue(i,j,k);
			}
		}
	}

/*---------------------------------------------------------------------------------------------------------------------------*/
	for(Index k = Mesh.getDimensions().z() -1; k > -1; k--)
	{
		for(Index i = 0; i < Mesh.getDimensions().x(); i++)
		{
			for(Index j = Mesh.getDimensions().y() - 1; j > -1; j--)
			{
				updateValue(i,j,k);
			}
		}
	}

/*---------------------------------------------------------------------------------------------------------------------------*/


	dofVector2.save("u-00001.tnl");

	cout << "bla 3"<<endl;
	return true;
}


template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
void tnlFastSweeping< tnlGrid< 3,MeshReal, Device, MeshIndex >, Real, Index > :: updateValue( Index i, Index j, Index k)
{
	this->Entity.setCoordinates(CoordinatesType(i,j,k));
	this->Entity.refresh();
	tnlNeighbourGridEntityGetter<tnlGridEntity< MeshType, 3, tnlGridEntityNoStencilStorage >,3> neighbourEntities(Entity);
	Real value = dofVector2[Entity.getIndex()];
	Real a,b,c, tmp;

	if( i == 0 )
		a = dofVector2[neighbourEntities.template getEntityIndex< 1,  0,  0>()];
	else if( i == Mesh.getDimensions().x() - 1 )
		a = dofVector2[neighbourEntities.template getEntityIndex< -1,  0,  0 >()];
	else
	{
		a = fabsMin( dofVector2[neighbourEntities.template getEntityIndex< -1,  0,  0>()],
				 dofVector2[neighbourEntities.template getEntityIndex< 1,  0,  0>()] );
	}

	if( j == 0 )
		b = dofVector2[neighbourEntities.template getEntityIndex< 0,  1,  0>()];
	else if( j == Mesh.getDimensions().y() - 1 )
		b = dofVector2[neighbourEntities.template getEntityIndex< 0,  -1,  0>()];
	else
	{
		b = fabsMin( dofVector2[neighbourEntities.template getEntityIndex< 0,  -1,  0>()],
				 dofVector2[neighbourEntities.template getEntityIndex< 0,  1,  0>()] );
	}

	if( k == 0 )
		c = dofVector2[neighbourEntities.template getEntityIndex< 0,  0,  1>()];
	else if( k == Mesh.getDimensions().z() - 1 )
		c = dofVector2[neighbourEntities.template getEntityIndex< 0,  0,  -1>()];
	else
	{
		c = fabsMin( dofVector2[neighbourEntities.template getEntityIndex< 0,  0,  -1>()],
				 dofVector2[neighbourEntities.template getEntityIndex< 0,  0,  1>()] );
	}

	Real hD = 3.0*h*h - 2.0*(a*a+b*b+c*c-a*b-a*c-b*c);

	if(hD < 0.0)
		tmp = fabsMin(a,fabsMin(b,c)) + sign(value)*h;
	else
		tmp = (1.0/3.0) * ( a + b + c + sign(value)*sqrt(hD) );


	dofVector2[Entity.getIndex()]  = fabsMin(value, tmp);
}


template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
Real tnlFastSweeping< tnlGrid< 3,MeshReal, Device, MeshIndex >, Real, Index > :: fabsMin( Real x, Real y)
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
