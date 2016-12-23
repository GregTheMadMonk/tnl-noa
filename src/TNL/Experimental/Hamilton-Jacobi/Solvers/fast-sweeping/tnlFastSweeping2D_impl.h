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
tnlFastSweeping< tnlGrid< 2,MeshReal, Device, MeshIndex >, Real, Index > :: tnlFastSweeping()
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
	dofVector2.load(initialCondition);

	h = Mesh.template getSpaceStepsProducts< 1, 0 >();
	Entity.refresh();

	const String& exact_input = parameters.getParameter< String >( "exact-input" );

	if(exact_input == "no")
		exactInput=false;
	else
		exactInput=true;

	cout << "a" << endl;
	return initGrid();
}


template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
bool tnlFastSweeping< tnlGrid< 2,MeshReal, Device, MeshIndex >, Real, Index > :: initGrid()
{

	tnlNeighbourGridEntityGetter<tnlGridEntity< MeshType, 2, tnlGridEntityNoStencilStorage >,2> neighbourEntities(Entity);
	for(int i=0; i< Mesh.getDimensions().x()*Mesh.getDimensions().x();i++)
	{
		dofVector2[i]=INT_MAX*sign(dofVector[i]);
	}

	for(int i = 0 ; i < Mesh.getDimensions().x()-1; i++)
	{
		for(int j = 0 ; j < Mesh.getDimensions().x()-1; j++)
			{
			this->Entity.setCoordinates(CoordinatesType(i,j));
			this->Entity.refresh();
			neighbourEntities.refresh(Mesh,Entity.getIndex());

				if(dofVector[this->Entity.getIndex()] > 0)
				{
					if(dofVector[neighbourEntities.template getEntityIndex< 1,  0 >()] > 0)
					{
						if(dofVector[neighbourEntities.template getEntityIndex< 0,  1 >()] > 0)
						{
							if(dofVector[neighbourEntities.template getEntityIndex< 1,  1 >()] > 0)
								setupSquare1111(i,j);
							else
								setupSquare1110(i,j);
						}
						else
						{
							if(dofVector[neighbourEntities.template getEntityIndex< 1,  1 >()] > 0)
								setupSquare1101(i,j);
							else
								setupSquare1100(i,j);
						}
					}
					else
					{
						if(dofVector[neighbourEntities.template getEntityIndex< 0,  1 >()] > 0)
						{
							if(dofVector[neighbourEntities.template getEntityIndex< 1,  1 >()] > 0)
								setupSquare1011(i,j);
							else
								setupSquare1010(i,j);
						}
						else
						{
							if(dofVector[neighbourEntities.template getEntityIndex< 1,  1 >()] > 0)
								setupSquare1001(i,j);
							else
								setupSquare1000(i,j);
						}
					}
				}
				else
				{
					if(dofVector[neighbourEntities.template getEntityIndex< 1,  0 >()] > 0)
					{
						if(dofVector[neighbourEntities.template getEntityIndex< 0,  1 >()] > 0)
						{
							if(dofVector[neighbourEntities.template getEntityIndex< 1,  1 >()] > 0)
								setupSquare0111(i,j);
							else
								setupSquare0110(i,j);
						}
						else
						{
							if(dofVector[neighbourEntities.template getEntityIndex< 1,  1 >()] > 0)
								setupSquare0101(i,j);
							else
								setupSquare0100(i,j);
						}
					}
					else
					{
						if(dofVector[neighbourEntities.template getEntityIndex< 0,  1 >()] > 0)
						{
							if(dofVector[neighbourEntities.template getEntityIndex< 1,  1 >()] > 0)
								setupSquare0011(i,j);
							else
								setupSquare0010(i,j);
						}
						else
						{
							if(dofVector[neighbourEntities.template getEntityIndex< 1,  1 >()] > 0)
								setupSquare0001(i,j);
							else
								setupSquare0000(i,j);
						}
					}
				}

			}
	}
	cout << "a" << endl;

//	Real tmp = 0.0;
//	Real ax=0.5/sqrt(2.0);
//
//	if(!exactInput)
//	{
//		for(Index i = 0; i < Mesh.getDimensions().x()*Mesh.getDimensions().y(); i++)
//				dofVector[i]=0.5*h*sign(dofVector[i]);
//	}
//
//
//	for(Index i = 1; i < Mesh.getDimensions().x()-1; i++)
//	{
//		for(Index j = 1; j < Mesh.getDimensions().y()-1; j++)
//		{
//			 tmp = sign(dofVector[Mesh.getCellIndex(CoordinatesType(i,j))]);
//
//			if(tmp == 0.0)
//			{}
//			else if(dofVector[Mesh.getCellIndex(CoordinatesType(i+1,j))]*tmp < 0.0 ||
//					dofVector[Mesh.getCellIndex(CoordinatesType(i-1,j))]*tmp < 0.0 ||
//					dofVector[Mesh.getCellIndex(CoordinatesType(i,j+1))]*tmp < 0.0 ||
//					dofVector[Mesh.getCellIndex(CoordinatesType(i,j-1))]*tmp < 0.0 )
//			{}
//			else
//				dofVector[Mesh.getCellIndex(CoordinatesType(i,j))] = tmp*INT_MAX;
//		}
//	}
//
//
//
//	for(int i = 1; i < Mesh.getDimensions().x()-1; i++)
//	{
//		Index j = 0;
//		tmp = sign(dofVector[Mesh.getCellIndex(CoordinatesType(i,j))]);
//
//
//		if(tmp == 0.0)
//		{}
//		else if(dofVector[Mesh.getCellIndex(CoordinatesType(i+1,j))]*tmp < 0.0 ||
//				dofVector[Mesh.getCellIndex(CoordinatesType(i-1,j))]*tmp < 0.0 ||
//				dofVector[Mesh.getCellIndex(CoordinatesType(i,j+1))]*tmp < 0.0 )
//		{}
//		else
//			dofVector[Mesh.getCellIndex(CoordinatesType(i,j))] = tmp*INT_MAX;
//	}
//
//	for(int i = 1; i < Mesh.getDimensions().x()-1; i++)
//	{
//		Index j = Mesh.getDimensions().y() - 1;
//		tmp = sign(dofVector[Mesh.getCellIndex(CoordinatesType(i,j))]);
//
//
//		if(tmp == 0.0)
//		{}
//		else if(dofVector[Mesh.getCellIndex(CoordinatesType(i+1,j))]*tmp < 0.0 ||
//				dofVector[Mesh.getCellIndex(CoordinatesType(i-1,j))]*tmp < 0.0 ||
//				dofVector[Mesh.getCellIndex(CoordinatesType(i,j-1))]*tmp < 0.0 )
//		{}
//		else
//			dofVector[Mesh.getCellIndex(CoordinatesType(i,j))] = tmp*INT_MAX;
//	}
//
//	for(int j = 1; j < Mesh.getDimensions().y()-1; j++)
//	{
//		Index i = 0;
//		tmp = sign(dofVector[Mesh.getCellIndex(CoordinatesType(i,j))]);
//
//
//		if(tmp == 0.0)
//		{}
//		else if(dofVector[Mesh.getCellIndex(CoordinatesType(i+1,j))]*tmp < 0.0 ||
//				dofVector[Mesh.getCellIndex(CoordinatesType(i,j+1))]*tmp < 0.0 ||
//				dofVector[Mesh.getCellIndex(CoordinatesType(i,j-1))]*tmp < 0.0 )
//		{}
//		else
//			dofVector[Mesh.getCellIndex(CoordinatesType(i,j))] = tmp*INT_MAX;
//	}
//
//	for(int j = 1; j < Mesh.getDimensions().y()-1; j++)
//	{
//		Index i = Mesh.getDimensions().x() - 1;
//		tmp = sign(dofVector[Mesh.getCellIndex(CoordinatesType(i,j))]);
//
//
//		if(tmp == 0.0)
//		{}
//		else if(dofVector[Mesh.getCellIndex(CoordinatesType(i-1,j))]*tmp < 0.0 ||
//				dofVector[Mesh.getCellIndex(CoordinatesType(i,j+1))]*tmp < 0.0 ||
//				dofVector[Mesh.getCellIndex(CoordinatesType(i,j-1))]*tmp < 0.0 )
//		{}
//		else
//			dofVector[Mesh.getCellIndex(CoordinatesType(i,j))] = tmp*INT_MAX;
//	}
//
//
//	Index i = Mesh.getDimensions().x() - 1;
//	Index j = Mesh.getDimensions().y() - 1;
//
//	tmp = sign(dofVector[Mesh.getCellIndex(CoordinatesType(i,j))]);
//	if(dofVector[Mesh.getCellIndex(CoordinatesType(i-1,j))]*tmp > 0.0 &&
//			dofVector[Mesh.getCellIndex(CoordinatesType(i,j-1))]*tmp > 0.0)
//
//		dofVector[Mesh.getCellIndex(CoordinatesType(i,j))] = tmp*INT_MAX;
//
//
//
//	j = 0;
//	tmp = sign(dofVector[Mesh.getCellIndex(CoordinatesType(i,j))]);
//	if(dofVector[Mesh.getCellIndex(CoordinatesType(i-1,j))]*tmp > 0.0 &&
//			dofVector[Mesh.getCellIndex(CoordinatesType(i,j+1))]*tmp > 0.0)
//
//		dofVector[Mesh.getCellIndex(CoordinatesType(i,j))] = tmp*INT_MAX;
//
//
//
//	i = 0;
//	j = Mesh.getDimensions().y() -1;
//	tmp = sign(dofVector[Mesh.getCellIndex(CoordinatesType(i,j))]);
//	if(dofVector[Mesh.getCellIndex(CoordinatesType(i+1,j))]*tmp > 0.0 &&
//			dofVector[Mesh.getCellIndex(CoordinatesType(i,j-1))]*tmp > 0.0)
//
//		dofVector[Mesh.getCellIndex(CoordinatesType(i,j))] = tmp*INT_MAX;
//
//
//
//	j = 0;
//	tmp = sign(dofVector[Mesh.getCellIndex(CoordinatesType(i,j))]);
//	if(dofVector[Mesh.getCellIndex(CoordinatesType(i+1,j))]*tmp > 0.0 &&
//			dofVector[Mesh.getCellIndex(CoordinatesType(i,j+1))]*tmp > 0.0)
//
//		dofVector[Mesh.getCellIndex(CoordinatesType(i,j))] = tmp*INT_MAX;

	//data.setLike(dofVector2.getData());
	//data=dofVector2.getData();
	//cout << data.getType() << endl;
	dofVector2.save("u-00000.tnl");
	//dofVector2.getData().save("u-00000.tnl");

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


//	data.setLike(dofVector2.getData());
//	data = dofVector2.getData();
//	cout << data.getType() << endl;
	dofVector2.save("u-00001.tnl");
	//dofVector2.getData().save("u-00001.tnl");

	return true;
}


template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
void tnlFastSweeping< tnlGrid< 2,MeshReal, Device, MeshIndex >, Real, Index > :: updateValue( Index i, Index j)
{

	this->Entity.setCoordinates(CoordinatesType(i,j));
	this->Entity.refresh();
	tnlNeighbourGridEntityGetter<tnlGridEntity< MeshType, 2, tnlGridEntityNoStencilStorage >,2> neighbourEntities(Entity);

	Real value = dofVector2[Entity.getIndex()];
	Real a,b, tmp;

	if( i == 0 )
		a = dofVector2[neighbourEntities.template getEntityIndex< 1,  0 >()];
	else if( i == Mesh.getDimensions().x() - 1 )
		a = dofVector2[neighbourEntities.template getEntityIndex< -1,  0 >()];
	else
	{
		a = fabsMin( dofVector2[neighbourEntities.template getEntityIndex< -1,  0 >()],
				 dofVector2[neighbourEntities.template getEntityIndex< 1,  0 >()] );
	}

	if( j == 0 )
		b = dofVector2[neighbourEntities.template getEntityIndex< 0,  1 >()];
	else if( j == Mesh.getDimensions().y() - 1 )
		b = dofVector2[neighbourEntities.template getEntityIndex< 0,  -1 >()];
	else
	{
		b = fabsMin( dofVector2[neighbourEntities.template getEntityIndex< 0,  -1 >()],
				 dofVector2[neighbourEntities.template getEntityIndex< 0,  1 >()] );
	}


	if(fabs(a-b) >= h)
		tmp = fabsMin(a,b) + sign(value)*h;
	else
		tmp = 0.5 * (a + b + sign(value)*sqrt(2.0 * h * h - (a - b) * (a - b) ) );


	dofVector2[Entity.getIndex()] = fabsMin(value, tmp);

//	if(dofVector2[Entity.getIndex()] > 1.0)
//		cout << value << "    " << tmp << " " << dofVector2[Entity.getIndex()] << endl;
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



template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
void tnlFastSweeping< tnlGrid< 2,MeshReal, Device, MeshIndex >, Real, Index > :: setupSquare1111( Index i, Index j)
{
//	this->Entity.setCoordinates(CoordinatesType(i,j));
//	this->Entity.refresh();
//	auto neighbourEntities =  Entity.getNeighbourEntities();
//	dofVector2[Entity.getIndex()]=fabsMin(INT_MAX,dofVector2[Entity.getIndex()]);
//	dofVector2[neighbourEntities.template getEntityIndex< 0,  1 >()]=fabsMin(INT_MAX,dofVector2[neighbourEntities.template getEntityIndex< 0,  1 >()]);
//	dofVector2[neighbourEntities.template getEntityIndex< 1,  1 >()]=fabsMin(INT_MAX,dofVector2[neighbourEntities.template getEntityIndex< 1,  1 >()]);
//	dofVector2[neighbourEntities.template getEntityIndex< 1,  0 >()]=fabsMin(INT_MAX,dofVector2[neighbourEntities.template getEntityIndex< 1,  0 >()]);

}


template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
void tnlFastSweeping< tnlGrid< 2,MeshReal, Device, MeshIndex >, Real, Index > :: setupSquare0000( Index i, Index j)
{
//	this->Entity.setCoordinates(CoordinatesType(i,j));
//	this->Entity.refresh();
//	auto neighbourEntities =  Entity.getNeighbourEntities();
//	dofVector2[Entity.getIndex()]=fabsMin(-INT_MAX,dofVector2[(Entity.getIndex())]);
//	dofVector2[neighbourEntities.template getEntityIndex< 0,  1 >()]=fabsMin(-INT_MAX,dofVector2[neighbourEntities.template getEntityIndex< 0,  1 >()]);
//	dofVector2[neighbourEntities.template getEntityIndex< 1,  1 >()]=fabsMin(-INT_MAX,dofVector2[neighbourEntities.template getEntityIndex< 1,  1 >()]);
//	dofVector2[neighbourEntities.template getEntityIndex< 1,  0 >()]=fabsMin(-INT_MAX,dofVector2[neighbourEntities.template getEntityIndex< 1,  0 >()]);

}


template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
void tnlFastSweeping< tnlGrid< 2,MeshReal, Device, MeshIndex >, Real, Index > :: setupSquare1110( Index i, Index j)
{
	this->Entity.setCoordinates(CoordinatesType(i,j));
	this->Entity.refresh();
	auto neighbourEntities =  Entity.getNeighbourEntities();
	Real al,be, a,b,c,s;
	al=abs(dofVector[neighbourEntities.template getEntityIndex< 1,  1 >()]/
			(dofVector[neighbourEntities.template getEntityIndex< 1,  0 >()]-
			 dofVector[neighbourEntities.template getEntityIndex< 1,  1 >()]));

	be=abs(dofVector[neighbourEntities.template getEntityIndex< 1,  1 >()]/
			(dofVector[neighbourEntities.template getEntityIndex< 0,  1 >()]-
			 dofVector[neighbourEntities.template getEntityIndex< 1,  1 >()]));

	a = be/al;
	b=1.0;
	c=-be;
	s= h/sqrt(a*a+b*b);


	dofVector2[Entity.getIndex()]=fabsMin(abs(a*1+b*1+c)*s,dofVector2[(Entity.getIndex())]);
	dofVector2[neighbourEntities.template getEntityIndex< 0,  1 >()]=fabsMin(abs(a*0+b*1+c)*s,dofVector2[neighbourEntities.template getEntityIndex< 0,  1 >()]);
	dofVector2[neighbourEntities.template getEntityIndex< 1,  1 >()]=fabsMin(-abs(a*0+b*0+c)*s,dofVector2[neighbourEntities.template getEntityIndex< 1,  1 >()]);
	dofVector2[neighbourEntities.template getEntityIndex< 1,  0 >()]=fabsMin(abs(a*1+b*0+c)*s,dofVector2[neighbourEntities.template getEntityIndex< 1,  0 >()]);

}

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
void tnlFastSweeping< tnlGrid< 2,MeshReal, Device, MeshIndex >, Real, Index > :: setupSquare1101( Index i, Index j)
{
	this->Entity.setCoordinates(CoordinatesType(i,j));
	this->Entity.refresh();
	auto neighbourEntities =  Entity.getNeighbourEntities();
	Real al,be, a,b,c,s;
	al=abs(dofVector[neighbourEntities.template getEntityIndex< 0,  1 >()]/
			(dofVector[neighbourEntities.template getEntityIndex< 1,  1 >()]-
			 dofVector[neighbourEntities.template getEntityIndex< 0,  1 >()]));

	be=abs(dofVector[neighbourEntities.template getEntityIndex< 0,  1 >()]/
			(dofVector[Entity.getIndex()]-
			 dofVector[neighbourEntities.template getEntityIndex< 0,  1 >()]));

	a = be/al;
	b=1.0;
	c=-be;
	s= h/sqrt(a*a+b*b);


	dofVector2[Entity.getIndex()]=fabsMin(abs(a*0+b*1+c)*s,dofVector2[(Entity.getIndex())]);
	dofVector2[neighbourEntities.template getEntityIndex< 0,  1 >()]=fabsMin(abs(a*0+b*0+c)*s,dofVector2[neighbourEntities.template getEntityIndex< 0,  1 >()]);
	dofVector2[neighbourEntities.template getEntityIndex< 1,  1 >()]=fabsMin(abs(a*1+b*0+c)*s,dofVector2[neighbourEntities.template getEntityIndex< 1,  1 >()]);
	dofVector2[neighbourEntities.template getEntityIndex< 1,  0 >()]=fabsMin(-abs(a*1+b*1+c)*s,dofVector2[neighbourEntities.template getEntityIndex< 1,  0 >()]);

}

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
void tnlFastSweeping< tnlGrid< 2,MeshReal, Device, MeshIndex >, Real, Index > :: setupSquare1011( Index i, Index j)
{
	this->Entity.setCoordinates(CoordinatesType(i,j));
	this->Entity.refresh();
	auto neighbourEntities =  Entity.getNeighbourEntities();
	Real al,be, a,b,c,s;
	al=abs(dofVector[neighbourEntities.template getEntityIndex< 1,  0 >()]/
			(dofVector[Entity.getIndex()]-
			 dofVector[neighbourEntities.template getEntityIndex< 1,  0 >()]));

	be=abs(dofVector[neighbourEntities.template getEntityIndex< 1,  0 >()]/
			(dofVector[neighbourEntities.template getEntityIndex< 1,  1 >()]-
			 dofVector[neighbourEntities.template getEntityIndex< 1,  0 >()]));

	a = be/al;
	b=1.0;
	c=-be;
	s= h/sqrt(a*a+b*b);


	dofVector2[Entity.getIndex()]=fabsMin(abs(a*1+b*0+c)*s,dofVector2[(Entity.getIndex())]);
	dofVector2[neighbourEntities.template getEntityIndex< 0,  1 >()]=fabsMin(-abs(a*1+b*1+c)*s,dofVector2[neighbourEntities.template getEntityIndex< 0,  1 >()]);
	dofVector2[neighbourEntities.template getEntityIndex< 1,  1 >()]=fabsMin(abs(a*0+b*1+c)*s,dofVector2[neighbourEntities.template getEntityIndex< 1,  1 >()]);
	dofVector2[neighbourEntities.template getEntityIndex< 1,  0 >()]=fabsMin(abs(a*0+b*0+c)*s,dofVector2[neighbourEntities.template getEntityIndex< 1,  0 >()]);

}

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
void tnlFastSweeping< tnlGrid< 2,MeshReal, Device, MeshIndex >, Real, Index > :: setupSquare0111( Index i, Index j)
{
	this->Entity.setCoordinates(CoordinatesType(i,j));
	this->Entity.refresh();
	auto neighbourEntities =  Entity.getNeighbourEntities();
	Real al,be, a,b,c,s;
	al=abs(dofVector[Entity.getIndex()]/
			(dofVector[neighbourEntities.template getEntityIndex< 0,  1 >()]-
			 dofVector[Entity.getIndex()]));

	be=abs(dofVector[Entity.getIndex()]/
			(dofVector[neighbourEntities.template getEntityIndex< 1,  0 >()]-
			 dofVector[Entity.getIndex()]));

	a = be/al;
	b=1.0;
	c=-be;
	s= h/sqrt(a*a+b*b);


	dofVector2[Entity.getIndex()]=fabsMin(-abs(a*0+b*0+c)*s,dofVector2[(Entity.getIndex())]);
	dofVector2[neighbourEntities.template getEntityIndex< 0,  1 >()]=fabsMin(abs(a*1+b*0+c)*s,dofVector2[neighbourEntities.template getEntityIndex< 0,  1 >()]);
	dofVector2[neighbourEntities.template getEntityIndex< 1,  1 >()]=fabsMin(abs(a*1+b*1+c)*s,dofVector2[neighbourEntities.template getEntityIndex< 1,  1 >()]);
	dofVector2[neighbourEntities.template getEntityIndex< 1,  0 >()]=fabsMin(abs(a*0+b*1+c)*s,dofVector2[neighbourEntities.template getEntityIndex< 1,  0 >()]);

}


template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
void tnlFastSweeping< tnlGrid< 2,MeshReal, Device, MeshIndex >, Real, Index > :: setupSquare0001( Index i, Index j)
{
	this->Entity.setCoordinates(CoordinatesType(i,j));
	this->Entity.refresh();
	auto neighbourEntities =  Entity.getNeighbourEntities();
	Real al,be, a,b,c,s;
	al=abs(dofVector[neighbourEntities.template getEntityIndex< 1,  1 >()]/
			(dofVector[neighbourEntities.template getEntityIndex< 1,  0 >()]-
			 dofVector[neighbourEntities.template getEntityIndex< 1,  1 >()]));

	be=abs(dofVector[neighbourEntities.template getEntityIndex< 1,  1 >()]/
			(dofVector[neighbourEntities.template getEntityIndex< 0,  1 >()]-
			 dofVector[neighbourEntities.template getEntityIndex< 1,  1 >()]));

	a = be/al;
	b=1.0;
	c=-be;
	s= h/sqrt(a*a+b*b);


	dofVector2[Entity.getIndex()]=fabsMin(-abs(a*1+b*1+c)*s,dofVector2[(Entity.getIndex())]);
	dofVector2[neighbourEntities.template getEntityIndex< 0,  1 >()]=fabsMin(-abs(a*0+b*1+c)*s,dofVector2[neighbourEntities.template getEntityIndex< 0,  1 >()]);
	dofVector2[neighbourEntities.template getEntityIndex< 1,  1 >()]=fabsMin(abs(a*0+b*0+c)*s,dofVector2[neighbourEntities.template getEntityIndex< 1,  1 >()]);
	dofVector2[neighbourEntities.template getEntityIndex< 1,  0 >()]=fabsMin(-abs(a*1+b*0+c)*s,dofVector2[neighbourEntities.template getEntityIndex< 1,  0 >()]);

}

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
void tnlFastSweeping< tnlGrid< 2,MeshReal, Device, MeshIndex >, Real, Index > :: setupSquare0010( Index i, Index j)
{
	this->Entity.setCoordinates(CoordinatesType(i,j));
	this->Entity.refresh();
	auto neighbourEntities =  Entity.getNeighbourEntities();
	Real al,be, a,b,c,s;
	al=abs(dofVector[neighbourEntities.template getEntityIndex< 0,  1 >()]/
			(dofVector[neighbourEntities.template getEntityIndex< 1,  1 >()]-
			 dofVector[neighbourEntities.template getEntityIndex< 0,  1 >()]));

	be=abs(dofVector[neighbourEntities.template getEntityIndex< 0,  1 >()]/
			(dofVector[Entity.getIndex()]-
			 dofVector[neighbourEntities.template getEntityIndex< 0,  1 >()]));

	a = be/al;
	b=1.0;
	c=-be;
	s= h/sqrt(a*a+b*b);


	dofVector2[Entity.getIndex()]=fabsMin(-abs(a*0+b*1+c)*s,dofVector2[(Entity.getIndex())]);
	dofVector2[neighbourEntities.template getEntityIndex< 0,  1 >()]=fabsMin(-abs(a*0+b*0+c)*s,dofVector2[neighbourEntities.template getEntityIndex< 0,  1 >()]);
	dofVector2[neighbourEntities.template getEntityIndex< 1,  1 >()]=fabsMin(-abs(a*1+b*0+c)*s,dofVector2[neighbourEntities.template getEntityIndex< 1,  1 >()]);
	dofVector2[neighbourEntities.template getEntityIndex< 1,  0 >()]=fabsMin(abs(a*1+b*1+c)*s,dofVector2[neighbourEntities.template getEntityIndex< 1,  0 >()]);

}

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
void tnlFastSweeping< tnlGrid< 2,MeshReal, Device, MeshIndex >, Real, Index > :: setupSquare0100( Index i, Index j)
{
	this->Entity.setCoordinates(CoordinatesType(i,j));
	this->Entity.refresh();
	auto neighbourEntities =  Entity.getNeighbourEntities();
	Real al,be, a,b,c,s;
	al=abs(dofVector[neighbourEntities.template getEntityIndex< 1,  0 >()]/
			(dofVector[Entity.getIndex()]-
			 dofVector[neighbourEntities.template getEntityIndex< 1,  0 >()]));

	be=abs(dofVector[neighbourEntities.template getEntityIndex< 1,  0 >()]/
			(dofVector[neighbourEntities.template getEntityIndex< 1,  1 >()]-
			 dofVector[neighbourEntities.template getEntityIndex< 1,  0 >()]));

	a = be/al;
	b=1.0;
	c=-be;
	s= h/sqrt(a*a+b*b);


	dofVector2[Entity.getIndex()]=fabsMin(-abs(a*1+b*0+c)*s,dofVector2[(Entity.getIndex())]);
	dofVector2[neighbourEntities.template getEntityIndex< 0,  1 >()]=fabsMin(abs(a*1+b*1+c)*s,dofVector2[neighbourEntities.template getEntityIndex< 0,  1 >()]);
	dofVector2[neighbourEntities.template getEntityIndex< 1,  1 >()]=fabsMin(-abs(a*0+b*1+c)*s,dofVector2[neighbourEntities.template getEntityIndex< 1,  1 >()]);
	dofVector2[neighbourEntities.template getEntityIndex< 1,  0 >()]=fabsMin(-abs(a*0+b*0+c)*s,dofVector2[neighbourEntities.template getEntityIndex< 1,  0 >()]);

}

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
void tnlFastSweeping< tnlGrid< 2,MeshReal, Device, MeshIndex >, Real, Index > :: setupSquare1000( Index i, Index j)
{
	this->Entity.setCoordinates(CoordinatesType(i,j));
	this->Entity.refresh();
	auto neighbourEntities =  Entity.getNeighbourEntities();
	Real al,be, a,b,c,s;
	al=abs(dofVector[Entity.getIndex()]/
			(dofVector[neighbourEntities.template getEntityIndex< 0,  1 >()]-
			 dofVector[Entity.getIndex()]));

	be=abs(dofVector[Entity.getIndex()]/
			(dofVector[neighbourEntities.template getEntityIndex< 1,  0 >()]-
			 dofVector[Entity.getIndex()]));

	a = be/al;
	b=1.0;
	c=-be;
	s= h/sqrt(a*a+b*b);


	dofVector2[Entity.getIndex()]=fabsMin(abs(a*0+b*0+c)*s,dofVector2[(Entity.getIndex())]);
	dofVector2[neighbourEntities.template getEntityIndex< 0,  1 >()]=fabsMin(-abs(a*1+b*0+c)*s,dofVector2[neighbourEntities.template getEntityIndex< 0,  1 >()]);
	dofVector2[neighbourEntities.template getEntityIndex< 1,  1 >()]=fabsMin(-abs(a*1+b*1+c)*s,dofVector2[neighbourEntities.template getEntityIndex< 1,  1 >()]);
	dofVector2[neighbourEntities.template getEntityIndex< 1,  0 >()]=fabsMin(-abs(a*0+b*1+c)*s,dofVector2[neighbourEntities.template getEntityIndex< 1,  0 >()]);

}





template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
void tnlFastSweeping< tnlGrid< 2,MeshReal, Device, MeshIndex >, Real, Index > :: setupSquare1100( Index i, Index j)
{
	this->Entity.setCoordinates(CoordinatesType(i,j));
	this->Entity.refresh();
	auto neighbourEntities =  Entity.getNeighbourEntities();
	Real al,be, a,b,c,s;
	al=abs(dofVector[Entity.getIndex()]/
			(dofVector[neighbourEntities.template getEntityIndex< 0,  1 >()]-
			 dofVector[Entity.getIndex()]));

	be=abs(dofVector[neighbourEntities.template getEntityIndex< 1,  0 >()]/
			(dofVector[neighbourEntities.template getEntityIndex< 1,  1 >()]-
			 dofVector[neighbourEntities.template getEntityIndex< 1,  0 >()]));

	a = al-be;
	b=1.0;
	c=-al;
	s= h/sqrt(a*a+b*b);


	dofVector2[Entity.getIndex()]=fabsMin(abs(a*0+b*0+c)*s,dofVector2[(Entity.getIndex())]);
	dofVector2[neighbourEntities.template getEntityIndex< 0,  1 >()]=fabsMin(-abs(a*0+b*1+c)*s,dofVector2[neighbourEntities.template getEntityIndex< 0,  1 >()]);
	dofVector2[neighbourEntities.template getEntityIndex< 1,  1 >()]=fabsMin(-abs(a*1+b*1+c)*s,dofVector2[neighbourEntities.template getEntityIndex< 1,  1 >()]);
	dofVector2[neighbourEntities.template getEntityIndex< 1,  0 >()]=fabsMin(abs(a*1+b*0+c)*s,dofVector2[neighbourEntities.template getEntityIndex< 1,  0 >()]);

}

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
void tnlFastSweeping< tnlGrid< 2,MeshReal, Device, MeshIndex >, Real, Index > :: setupSquare1010( Index i, Index j)
{
	this->Entity.setCoordinates(CoordinatesType(i,j));
	this->Entity.refresh();
	auto neighbourEntities =  Entity.getNeighbourEntities();
	Real al,be, a,b,c,s;
	al=abs(dofVector[Entity.getIndex()]/
			(dofVector[neighbourEntities.template getEntityIndex< 1,  0 >()]-
			 dofVector[Entity.getIndex()]));

	be=abs(dofVector[neighbourEntities.template getEntityIndex< 0,  1 >()]/
			(dofVector[neighbourEntities.template getEntityIndex< 1,  1 >()]-
			 dofVector[neighbourEntities.template getEntityIndex< 0,  1 >()]));

	a = al-be;
	b=1.0;
	c=-be;
	s= h/sqrt(a*a+b*b);


	dofVector2[Entity.getIndex()]=fabsMin(abs(a*0+b*0+c)*s,dofVector2[(Entity.getIndex())]);
	dofVector2[neighbourEntities.template getEntityIndex< 0,  1 >()]=fabsMin(abs(a*1+b*0+c)*s,dofVector2[neighbourEntities.template getEntityIndex< 0,  1 >()]);
	dofVector2[neighbourEntities.template getEntityIndex< 1,  1 >()]=fabsMin(-abs(a*1+b*1+c)*s,dofVector2[neighbourEntities.template getEntityIndex< 1,  1 >()]);
	dofVector2[neighbourEntities.template getEntityIndex< 1,  0 >()]=fabsMin(-abs(a*0+b*1+c)*s,dofVector2[neighbourEntities.template getEntityIndex< 1,  0 >()]);

}

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
void tnlFastSweeping< tnlGrid< 2,MeshReal, Device, MeshIndex >, Real, Index > :: setupSquare1001( Index i, Index j)
{
	this->Entity.setCoordinates(CoordinatesType(i,j));
	this->Entity.refresh();
	auto neighbourEntities =  Entity.getNeighbourEntities();
	dofVector2[Entity.getIndex()]=fabsMin(dofVector[Entity.getIndex()],dofVector2[(Entity.getIndex())]);
	dofVector2[neighbourEntities.template getEntityIndex< 0,  1 >()]=fabsMin(dofVector[neighbourEntities.template getEntityIndex< 0,  1 >()],dofVector2[neighbourEntities.template getEntityIndex< 0,  1 >()]);
	dofVector2[neighbourEntities.template getEntityIndex< 1,  1 >()]=fabsMin(dofVector[neighbourEntities.template getEntityIndex< 1,  1 >()],dofVector2[neighbourEntities.template getEntityIndex< 1,  1 >()]);
	dofVector2[neighbourEntities.template getEntityIndex< 1,  0 >()]=fabsMin(dofVector[neighbourEntities.template getEntityIndex< 1,  0 >()],dofVector2[neighbourEntities.template getEntityIndex< 1,  0 >()]);

}







template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
void tnlFastSweeping< tnlGrid< 2,MeshReal, Device, MeshIndex >, Real, Index > :: setupSquare0011( Index i, Index j)
{
	this->Entity.setCoordinates(CoordinatesType(i,j));
	this->Entity.refresh();
	auto neighbourEntities =  Entity.getNeighbourEntities();
	Real al,be, a,b,c,s;
	al=abs(dofVector[Entity.getIndex()]/
			(dofVector[neighbourEntities.template getEntityIndex< 0,  1 >()]-
			 dofVector[Entity.getIndex()]));

	be=abs(dofVector[neighbourEntities.template getEntityIndex< 1,  0 >()]/
			(dofVector[neighbourEntities.template getEntityIndex< 1,  1 >()]-
			 dofVector[neighbourEntities.template getEntityIndex< 1,  0 >()]));

	a = al-be;
	b=1.0;
	c=-al;
	s= h/sqrt(a*a+b*b);


	dofVector2[Entity.getIndex()]=fabsMin(-abs(a*0+b*0+c)*s,dofVector2[(Entity.getIndex())]);
	dofVector2[neighbourEntities.template getEntityIndex< 0,  1 >()]=fabsMin(abs(a*0+b*1+c)*s,dofVector2[neighbourEntities.template getEntityIndex< 0,  1 >()]);
	dofVector2[neighbourEntities.template getEntityIndex< 1,  1 >()]=fabsMin(abs(a*1+b*1+c)*s,dofVector2[neighbourEntities.template getEntityIndex< 1,  1 >()]);
	dofVector2[neighbourEntities.template getEntityIndex< 1,  0 >()]=fabsMin(-abs(a*1+b*0+c)*s,dofVector2[neighbourEntities.template getEntityIndex< 1,  0 >()]);

}

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
void tnlFastSweeping< tnlGrid< 2,MeshReal, Device, MeshIndex >, Real, Index > :: setupSquare0101( Index i, Index j)
{
	this->Entity.setCoordinates(CoordinatesType(i,j));
	this->Entity.refresh();
	auto neighbourEntities =  Entity.getNeighbourEntities();
	Real al,be, a,b,c,s;
	al=abs(dofVector[Entity.getIndex()]/
			(dofVector[neighbourEntities.template getEntityIndex< 1,  0 >()]-
			 dofVector[Entity.getIndex()]));

	be=abs(dofVector[neighbourEntities.template getEntityIndex< 0,  1 >()]/
			(dofVector[neighbourEntities.template getEntityIndex< 1,  1 >()]-
			 dofVector[neighbourEntities.template getEntityIndex< 0,  1 >()]));

	a = al-be;
	b=1.0;
	c=-be;
	s= h/sqrt(a*a+b*b);


	dofVector2[Entity.getIndex()]=fabsMin(-abs(a*0+b*0+c)*s,dofVector2[(Entity.getIndex())]);
	dofVector2[neighbourEntities.template getEntityIndex< 0,  1 >()]=fabsMin(-abs(a*1+b*0+c)*s,dofVector2[neighbourEntities.template getEntityIndex< 0,  1 >()]);
	dofVector2[neighbourEntities.template getEntityIndex< 1,  1 >()]=fabsMin(abs(a*1+b*1+c)*s,dofVector2[neighbourEntities.template getEntityIndex< 1,  1 >()]);
	dofVector2[neighbourEntities.template getEntityIndex< 1,  0 >()]=fabsMin(abs(a*0+b*1+c)*s,dofVector2[neighbourEntities.template getEntityIndex< 1,  0 >()]);

}

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
void tnlFastSweeping< tnlGrid< 2,MeshReal, Device, MeshIndex >, Real, Index > :: setupSquare0110( Index i, Index j)
{
	this->Entity.setCoordinates(CoordinatesType(i,j));
	this->Entity.refresh();
	auto neighbourEntities =  Entity.getNeighbourEntities();
	dofVector2[Entity.getIndex()]=fabsMin(dofVector[Entity.getIndex()],dofVector2[(Entity.getIndex())]);
	dofVector2[neighbourEntities.template getEntityIndex< 0,  1 >()]=fabsMin(dofVector[neighbourEntities.template getEntityIndex< 0,  1 >()],dofVector2[neighbourEntities.template getEntityIndex< 0,  1 >()]);
	dofVector2[neighbourEntities.template getEntityIndex< 1,  1 >()]=fabsMin(dofVector[neighbourEntities.template getEntityIndex< 1,  1 >()],dofVector2[neighbourEntities.template getEntityIndex< 1,  1 >()]);
	dofVector2[neighbourEntities.template getEntityIndex< 1,  0 >()]=fabsMin(dofVector[neighbourEntities.template getEntityIndex< 1,  0 >()],dofVector2[neighbourEntities.template getEntityIndex< 1,  0 >()]);
}




#endif /* TNLFASTSWEEPING_IMPL_H_ */
