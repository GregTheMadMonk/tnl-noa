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
tnlString tnlFastSweeping< tnlGrid< 3,MeshReal, Device, MeshIndex >, Real, Index > :: getType()
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
bool tnlFastSweeping< tnlGrid< 3,MeshReal, Device, MeshIndex >, Real, Index > :: init( const tnlParameterContainer& parameters )
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
	dofVector2.setLike(dofVector);

	h = Mesh.getHx();

	const tnlString& exact_input = parameters.getParameter< tnlString >( "exact-input" );

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
			dofVector2[i]=INT_MAX*Sign(dofVector[i]);
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
	Index index = Mesh.getCellIndex(CoordinatesType(i,j,k));
	Real value = dofVector2[index];
	Real a,b,c, tmp;

	if( i == 0 )
		a = dofVector2[Mesh.template getCellNextToCell<1,0,0>(index)];
	else if( i == Mesh.getDimensions().x() - 1 )
		a = dofVector2[Mesh.template getCellNextToCell<-1,0,0>(index)];
	else
	{
		a = fabsMin( dofVector2[Mesh.template getCellNextToCell<-1,0,0>(index)],
				 dofVector2[Mesh.template getCellNextToCell<1,0,0>(index)] );
	}

	if( j == 0 )
		b = dofVector2[Mesh.template getCellNextToCell<0,1,0>(index)];
	else if( j == Mesh.getDimensions().y() - 1 )
		b = dofVector2[Mesh.template getCellNextToCell<0,-1,0>(index)];
	else
	{
		b = fabsMin( dofVector2[Mesh.template getCellNextToCell<0,-1,0>(index)],
				 dofVector2[Mesh.template getCellNextToCell<0,1,0>(index)] );
	}

	if( k == 0 )
		c = dofVector2[Mesh.template getCellNextToCell<0,0,1>(index)];
	else if( k == Mesh.getDimensions().z() - 1 )
		c = dofVector2[Mesh.template getCellNextToCell<0,0,-1>(index)];
	else
	{
		c = fabsMin( dofVector2[Mesh.template getCellNextToCell<0,0,-1>(index)],
				 dofVector2[Mesh.template getCellNextToCell<0,0,1>(index)] );
	}

	Real hD = 3.0*h*h - 2.0*(a*a+b*b+c*c-a*b-a*c-b*c);

	if(hD < 0.0)
		tmp = fabsMin(a,fabsMin(b,c)) + Sign(value)*h;
	else
		tmp = (1.0/3.0) * ( a + b + c + Sign(value)*sqrt(hD) );


	dofVector2[index]  = fabsMin(value, tmp);
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



template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
void tnlFastSweeping< tnlGrid< 3,MeshReal, Device, MeshIndex >, Real, Index > :: setupSquare1111( Index i, Index j)
{
	Index index = Mesh.getCellIndex(CoordinatesType(i,j));
	dofVector2[index]=fabsMin(INT_MAX,dofVector2[(index)]);
	dofVector2[Mesh.template getCellNextToCell<0,1>(index)]=fabsMin(INT_MAX,dofVector2[Mesh.template getCellNextToCell<0,1>(index)]);
	dofVector2[Mesh.template getCellNextToCell<1,1>(index)]=fabsMin(INT_MAX,dofVector2[Mesh.template getCellNextToCell<1,1>(index)]);
	dofVector2[Mesh.template getCellNextToCell<1,0>(index)]=fabsMin(INT_MAX,dofVector2[Mesh.template getCellNextToCell<1,0>(index)]);

}


template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
void tnlFastSweeping< tnlGrid< 3,MeshReal, Device, MeshIndex >, Real, Index > :: setupSquare0000( Index i, Index j)
{
	Index index = Mesh.getCellIndex(CoordinatesType(i,j));
	dofVector2[index]=fabsMin(-INT_MAX,dofVector2[(index)]);
	dofVector2[Mesh.template getCellNextToCell<0,1>(index)]=fabsMin(-INT_MAX,dofVector2[Mesh.template getCellNextToCell<0,1>(index)]);
	dofVector2[Mesh.template getCellNextToCell<1,1>(index)]=fabsMin(-INT_MAX,dofVector2[Mesh.template getCellNextToCell<1,1>(index)]);
	dofVector2[Mesh.template getCellNextToCell<1,0>(index)]=fabsMin(-INT_MAX,dofVector2[Mesh.template getCellNextToCell<1,0>(index)]);

}


template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
void tnlFastSweeping< tnlGrid< 3,MeshReal, Device, MeshIndex >, Real, Index > :: setupSquare1110( Index i, Index j)
{
	Index index = Mesh.getCellIndex(CoordinatesType(i,j));
	Real al,be, a,b,c,s;
	al=abs(dofVector[Mesh.template getCellNextToCell<1,1>(index)]/
			(dofVector[Mesh.template getCellNextToCell<1,0>(index)]-
			 dofVector[Mesh.template getCellNextToCell<1,1>(index)]));

	be=abs(dofVector[Mesh.template getCellNextToCell<1,1>(index)]/
			(dofVector[Mesh.template getCellNextToCell<0,1>(index)]-
			 dofVector[Mesh.template getCellNextToCell<1,1>(index)]));

	a = be/al;
	b=1.0;
	c=-be;
	s= h/sqrt(a*a+b*b);


	dofVector2[index]=fabsMin(abs(a*1+b*1+c)*s,dofVector2[(index)]);
	dofVector2[Mesh.template getCellNextToCell<0,1>(index)]=fabsMin(abs(a*0+b*1+c)*s,dofVector2[Mesh.template getCellNextToCell<0,1>(index)]);
	dofVector2[Mesh.template getCellNextToCell<1,1>(index)]=fabsMin(-abs(a*0+b*0+c)*s,dofVector2[Mesh.template getCellNextToCell<1,1>(index)]);
	dofVector2[Mesh.template getCellNextToCell<1,0>(index)]=fabsMin(abs(a*1+b*0+c)*s,dofVector2[Mesh.template getCellNextToCell<1,0>(index)]);

}

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
void tnlFastSweeping< tnlGrid< 3,MeshReal, Device, MeshIndex >, Real, Index > :: setupSquare1101( Index i, Index j)
{
	Index index = Mesh.getCellIndex(CoordinatesType(i,j));
	Real al,be, a,b,c,s;
	al=abs(dofVector[Mesh.template getCellNextToCell<0,1>(index)]/
			(dofVector[Mesh.template getCellNextToCell<1,1>(index)]-
			 dofVector[Mesh.template getCellNextToCell<0,1>(index)]));

	be=abs(dofVector[Mesh.template getCellNextToCell<0,1>(index)]/
			(dofVector[Mesh.template getCellNextToCell<0,0>(index)]-
			 dofVector[Mesh.template getCellNextToCell<0,1>(index)]));

	a = be/al;
	b=1.0;
	c=-be;
	s= h/sqrt(a*a+b*b);


	dofVector2[index]=fabsMin(abs(a*0+b*1+c)*s,dofVector2[(index)]);
	dofVector2[Mesh.template getCellNextToCell<0,1>(index)]=fabsMin(abs(a*0+b*0+c)*s,dofVector2[Mesh.template getCellNextToCell<0,1>(index)]);
	dofVector2[Mesh.template getCellNextToCell<1,1>(index)]=fabsMin(abs(a*1+b*0+c)*s,dofVector2[Mesh.template getCellNextToCell<1,1>(index)]);
	dofVector2[Mesh.template getCellNextToCell<1,0>(index)]=fabsMin(-abs(a*1+b*1+c)*s,dofVector2[Mesh.template getCellNextToCell<1,0>(index)]);

}

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
void tnlFastSweeping< tnlGrid< 3,MeshReal, Device, MeshIndex >, Real, Index > :: setupSquare1011( Index i, Index j)
{
	Index index = Mesh.getCellIndex(CoordinatesType(i,j));
	Real al,be, a,b,c,s;
	al=abs(dofVector[Mesh.template getCellNextToCell<1,0>(index)]/
			(dofVector[Mesh.template getCellNextToCell<0,0>(index)]-
			 dofVector[Mesh.template getCellNextToCell<1,0>(index)]));

	be=abs(dofVector[Mesh.template getCellNextToCell<1,0>(index)]/
			(dofVector[Mesh.template getCellNextToCell<1,1>(index)]-
			 dofVector[Mesh.template getCellNextToCell<1,0>(index)]));

	a = be/al;
	b=1.0;
	c=-be;
	s= h/sqrt(a*a+b*b);


	dofVector2[index]=fabsMin(abs(a*1+b*0+c)*s,dofVector2[(index)]);
	dofVector2[Mesh.template getCellNextToCell<0,1>(index)]=fabsMin(-abs(a*1+b*1+c)*s,dofVector2[Mesh.template getCellNextToCell<0,1>(index)]);
	dofVector2[Mesh.template getCellNextToCell<1,1>(index)]=fabsMin(abs(a*0+b*1+c)*s,dofVector2[Mesh.template getCellNextToCell<1,1>(index)]);
	dofVector2[Mesh.template getCellNextToCell<1,0>(index)]=fabsMin(abs(a*0+b*0+c)*s,dofVector2[Mesh.template getCellNextToCell<1,0>(index)]);

}

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
void tnlFastSweeping< tnlGrid< 3,MeshReal, Device, MeshIndex >, Real, Index > :: setupSquare0111( Index i, Index j)
{
	Index index = Mesh.getCellIndex(CoordinatesType(i,j));
	Real al,be, a,b,c,s;
	al=abs(dofVector[Mesh.template getCellNextToCell<0,0>(index)]/
			(dofVector[Mesh.template getCellNextToCell<0,1>(index)]-
			 dofVector[Mesh.template getCellNextToCell<0,0>(index)]));

	be=abs(dofVector[Mesh.template getCellNextToCell<0,0>(index)]/
			(dofVector[Mesh.template getCellNextToCell<1,0>(index)]-
			 dofVector[Mesh.template getCellNextToCell<0,0>(index)]));

	a = be/al;
	b=1.0;
	c=-be;
	s= h/sqrt(a*a+b*b);


	dofVector2[index]=fabsMin(-abs(a*0+b*0+c)*s,dofVector2[(index)]);
	dofVector2[Mesh.template getCellNextToCell<0,1>(index)]=fabsMin(abs(a*1+b*0+c)*s,dofVector2[Mesh.template getCellNextToCell<0,1>(index)]);
	dofVector2[Mesh.template getCellNextToCell<1,1>(index)]=fabsMin(abs(a*1+b*1+c)*s,dofVector2[Mesh.template getCellNextToCell<1,1>(index)]);
	dofVector2[Mesh.template getCellNextToCell<1,0>(index)]=fabsMin(abs(a*0+b*1+c)*s,dofVector2[Mesh.template getCellNextToCell<1,0>(index)]);

}


template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
void tnlFastSweeping< tnlGrid< 3,MeshReal, Device, MeshIndex >, Real, Index > :: setupSquare0001( Index i, Index j)
{
	Index index = Mesh.getCellIndex(CoordinatesType(i,j));
	Real al,be, a,b,c,s;
	al=abs(dofVector[Mesh.template getCellNextToCell<1,1>(index)]/
			(dofVector[Mesh.template getCellNextToCell<1,0>(index)]-
			 dofVector[Mesh.template getCellNextToCell<1,1>(index)]));

	be=abs(dofVector[Mesh.template getCellNextToCell<1,1>(index)]/
			(dofVector[Mesh.template getCellNextToCell<0,1>(index)]-
			 dofVector[Mesh.template getCellNextToCell<1,1>(index)]));

	a = be/al;
	b=1.0;
	c=-be;
	s= h/sqrt(a*a+b*b);


	dofVector2[index]=fabsMin(-abs(a*1+b*1+c)*s,dofVector2[(index)]);
	dofVector2[Mesh.template getCellNextToCell<0,1>(index)]=fabsMin(-abs(a*0+b*1+c)*s,dofVector2[Mesh.template getCellNextToCell<0,1>(index)]);
	dofVector2[Mesh.template getCellNextToCell<1,1>(index)]=fabsMin(abs(a*0+b*0+c)*s,dofVector2[Mesh.template getCellNextToCell<1,1>(index)]);
	dofVector2[Mesh.template getCellNextToCell<1,0>(index)]=fabsMin(-abs(a*1+b*0+c)*s,dofVector2[Mesh.template getCellNextToCell<1,0>(index)]);

}

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
void tnlFastSweeping< tnlGrid< 3,MeshReal, Device, MeshIndex >, Real, Index > :: setupSquare0010( Index i, Index j)
{
	Index index = Mesh.getCellIndex(CoordinatesType(i,j));
	Real al,be, a,b,c,s;
	al=abs(dofVector[Mesh.template getCellNextToCell<0,1>(index)]/
			(dofVector[Mesh.template getCellNextToCell<1,1>(index)]-
			 dofVector[Mesh.template getCellNextToCell<0,1>(index)]));

	be=abs(dofVector[Mesh.template getCellNextToCell<0,1>(index)]/
			(dofVector[Mesh.template getCellNextToCell<0,0>(index)]-
			 dofVector[Mesh.template getCellNextToCell<0,1>(index)]));

	a = be/al;
	b=1.0;
	c=-be;
	s= h/sqrt(a*a+b*b);


	dofVector2[index]=fabsMin(-abs(a*0+b*1+c)*s,dofVector2[(index)]);
	dofVector2[Mesh.template getCellNextToCell<0,1>(index)]=fabsMin(-abs(a*0+b*0+c)*s,dofVector2[Mesh.template getCellNextToCell<0,1>(index)]);
	dofVector2[Mesh.template getCellNextToCell<1,1>(index)]=fabsMin(-abs(a*1+b*0+c)*s,dofVector2[Mesh.template getCellNextToCell<1,1>(index)]);
	dofVector2[Mesh.template getCellNextToCell<1,0>(index)]=fabsMin(abs(a*1+b*1+c)*s,dofVector2[Mesh.template getCellNextToCell<1,0>(index)]);

}

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
void tnlFastSweeping< tnlGrid< 3,MeshReal, Device, MeshIndex >, Real, Index > :: setupSquare0100( Index i, Index j)
{
	Index index = Mesh.getCellIndex(CoordinatesType(i,j));
	Real al,be, a,b,c,s;
	al=abs(dofVector[Mesh.template getCellNextToCell<1,0>(index)]/
			(dofVector[Mesh.template getCellNextToCell<0,0>(index)]-
			 dofVector[Mesh.template getCellNextToCell<1,0>(index)]));

	be=abs(dofVector[Mesh.template getCellNextToCell<1,0>(index)]/
			(dofVector[Mesh.template getCellNextToCell<1,1>(index)]-
			 dofVector[Mesh.template getCellNextToCell<1,0>(index)]));

	a = be/al;
	b=1.0;
	c=-be;
	s= h/sqrt(a*a+b*b);


	dofVector2[index]=fabsMin(-abs(a*1+b*0+c)*s,dofVector2[(index)]);
	dofVector2[Mesh.template getCellNextToCell<0,1>(index)]=fabsMin(abs(a*1+b*1+c)*s,dofVector2[Mesh.template getCellNextToCell<0,1>(index)]);
	dofVector2[Mesh.template getCellNextToCell<1,1>(index)]=fabsMin(-abs(a*0+b*1+c)*s,dofVector2[Mesh.template getCellNextToCell<1,1>(index)]);
	dofVector2[Mesh.template getCellNextToCell<1,0>(index)]=fabsMin(-abs(a*0+b*0+c)*s,dofVector2[Mesh.template getCellNextToCell<1,0>(index)]);

}

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
void tnlFastSweeping< tnlGrid< 3,MeshReal, Device, MeshIndex >, Real, Index > :: setupSquare1000( Index i, Index j)
{
	Index index = Mesh.getCellIndex(CoordinatesType(i,j));
	Real al,be, a,b,c,s;
	al=abs(dofVector[Mesh.template getCellNextToCell<0,0>(index)]/
			(dofVector[Mesh.template getCellNextToCell<0,1>(index)]-
			 dofVector[Mesh.template getCellNextToCell<0,0>(index)]));

	be=abs(dofVector[Mesh.template getCellNextToCell<0,0>(index)]/
			(dofVector[Mesh.template getCellNextToCell<1,0>(index)]-
			 dofVector[Mesh.template getCellNextToCell<0,0>(index)]));

	a = be/al;
	b=1.0;
	c=-be;
	s= h/sqrt(a*a+b*b);


	dofVector2[index]=fabsMin(abs(a*0+b*0+c)*s,dofVector2[(index)]);
	dofVector2[Mesh.template getCellNextToCell<0,1>(index)]=fabsMin(-abs(a*1+b*0+c)*s,dofVector2[Mesh.template getCellNextToCell<0,1>(index)]);
	dofVector2[Mesh.template getCellNextToCell<1,1>(index)]=fabsMin(-abs(a*1+b*1+c)*s,dofVector2[Mesh.template getCellNextToCell<1,1>(index)]);
	dofVector2[Mesh.template getCellNextToCell<1,0>(index)]=fabsMin(-abs(a*0+b*1+c)*s,dofVector2[Mesh.template getCellNextToCell<1,0>(index)]);

}





template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
void tnlFastSweeping< tnlGrid< 3,MeshReal, Device, MeshIndex >, Real, Index > :: setupSquare1100( Index i, Index j)
{
	Index index = Mesh.getCellIndex(CoordinatesType(i,j));
	Real al,be, a,b,c,s;
	al=abs(dofVector[Mesh.template getCellNextToCell<0,0>(index)]/
			(dofVector[Mesh.template getCellNextToCell<0,1>(index)]-
			 dofVector[Mesh.template getCellNextToCell<0,0>(index)]));

	be=abs(dofVector[Mesh.template getCellNextToCell<1,0>(index)]/
			(dofVector[Mesh.template getCellNextToCell<1,1>(index)]-
			 dofVector[Mesh.template getCellNextToCell<1,0>(index)]));

	a = al-be;
	b=1.0;
	c=-al;
	s= h/sqrt(a*a+b*b);


	dofVector2[index]=fabsMin(abs(a*0+b*0+c)*s,dofVector2[(index)]);
	dofVector2[Mesh.template getCellNextToCell<0,1>(index)]=fabsMin(-abs(a*0+b*1+c)*s,dofVector2[Mesh.template getCellNextToCell<0,1>(index)]);
	dofVector2[Mesh.template getCellNextToCell<1,1>(index)]=fabsMin(-abs(a*1+b*1+c)*s,dofVector2[Mesh.template getCellNextToCell<1,1>(index)]);
	dofVector2[Mesh.template getCellNextToCell<1,0>(index)]=fabsMin(abs(a*1+b*0+c)*s,dofVector2[Mesh.template getCellNextToCell<1,0>(index)]);

}

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
void tnlFastSweeping< tnlGrid< 3,MeshReal, Device, MeshIndex >, Real, Index > :: setupSquare1010( Index i, Index j)
{
	Index index = Mesh.getCellIndex(CoordinatesType(i,j));
	Real al,be, a,b,c,s;
	al=abs(dofVector[Mesh.template getCellNextToCell<0,0>(index)]/
			(dofVector[Mesh.template getCellNextToCell<1,0>(index)]-
			 dofVector[Mesh.template getCellNextToCell<0,0>(index)]));

	be=abs(dofVector[Mesh.template getCellNextToCell<0,1>(index)]/
			(dofVector[Mesh.template getCellNextToCell<1,1>(index)]-
			 dofVector[Mesh.template getCellNextToCell<0,1>(index)]));

	a = al-be;
	b=1.0;
	c=-be;
	s= h/sqrt(a*a+b*b);


	dofVector2[index]=fabsMin(abs(a*0+b*0+c)*s,dofVector2[(index)]);
	dofVector2[Mesh.template getCellNextToCell<0,1>(index)]=fabsMin(abs(a*1+b*0+c)*s,dofVector2[Mesh.template getCellNextToCell<0,1>(index)]);
	dofVector2[Mesh.template getCellNextToCell<1,1>(index)]=fabsMin(-abs(a*1+b*1+c)*s,dofVector2[Mesh.template getCellNextToCell<1,1>(index)]);
	dofVector2[Mesh.template getCellNextToCell<1,0>(index)]=fabsMin(-abs(a*0+b*1+c)*s,dofVector2[Mesh.template getCellNextToCell<1,0>(index)]);

}

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
void tnlFastSweeping< tnlGrid< 3,MeshReal, Device, MeshIndex >, Real, Index > :: setupSquare1001( Index i, Index j)
{
	Index index = Mesh.getCellIndex(CoordinatesType(i,j));
	dofVector2[index]=fabsMin(dofVector[index],dofVector2[(index)]);
	dofVector2[Mesh.template getCellNextToCell<0,1>(index)]=fabsMin(dofVector[Mesh.template getCellNextToCell<0,1>(index)],dofVector2[Mesh.template getCellNextToCell<0,1>(index)]);
	dofVector2[Mesh.template getCellNextToCell<1,1>(index)]=fabsMin(dofVector[Mesh.template getCellNextToCell<1,1>(index)],dofVector2[Mesh.template getCellNextToCell<1,1>(index)]);
	dofVector2[Mesh.template getCellNextToCell<1,0>(index)]=fabsMin(dofVector[Mesh.template getCellNextToCell<1,0>(index)],dofVector2[Mesh.template getCellNextToCell<1,0>(index)]);

}







template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
void tnlFastSweeping< tnlGrid< 3,MeshReal, Device, MeshIndex >, Real, Index > :: setupSquare0011( Index i, Index j)
{
	Index index = Mesh.getCellIndex(CoordinatesType(i,j));
	Real al,be, a,b,c,s;
	al=abs(dofVector[Mesh.template getCellNextToCell<0,0>(index)]/
			(dofVector[Mesh.template getCellNextToCell<0,1>(index)]-
			 dofVector[Mesh.template getCellNextToCell<0,0>(index)]));

	be=abs(dofVector[Mesh.template getCellNextToCell<1,0>(index)]/
			(dofVector[Mesh.template getCellNextToCell<1,1>(index)]-
			 dofVector[Mesh.template getCellNextToCell<1,0>(index)]));

	a = al-be;
	b=1.0;
	c=-al;
	s= h/sqrt(a*a+b*b);


	dofVector2[index]=fabsMin(-abs(a*0+b*0+c)*s,dofVector2[(index)]);
	dofVector2[Mesh.template getCellNextToCell<0,1>(index)]=fabsMin(abs(a*0+b*1+c)*s,dofVector2[Mesh.template getCellNextToCell<0,1>(index)]);
	dofVector2[Mesh.template getCellNextToCell<1,1>(index)]=fabsMin(abs(a*1+b*1+c)*s,dofVector2[Mesh.template getCellNextToCell<1,1>(index)]);
	dofVector2[Mesh.template getCellNextToCell<1,0>(index)]=fabsMin(-abs(a*1+b*0+c)*s,dofVector2[Mesh.template getCellNextToCell<1,0>(index)]);

}

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
void tnlFastSweeping< tnlGrid< 3,MeshReal, Device, MeshIndex >, Real, Index > :: setupSquare0101( Index i, Index j)
{
	Index index = Mesh.getCellIndex(CoordinatesType(i,j));
	Real al,be, a,b,c,s;
	al=abs(dofVector[Mesh.template getCellNextToCell<0,0>(index)]/
			(dofVector[Mesh.template getCellNextToCell<1,0>(index)]-
			 dofVector[Mesh.template getCellNextToCell<0,0>(index)]));

	be=abs(dofVector[Mesh.template getCellNextToCell<0,1>(index)]/
			(dofVector[Mesh.template getCellNextToCell<1,1>(index)]-
			 dofVector[Mesh.template getCellNextToCell<0,1>(index)]));

	a = al-be;
	b=1.0;
	c=-be;
	s= h/sqrt(a*a+b*b);


	dofVector2[index]=fabsMin(-abs(a*0+b*0+c)*s,dofVector2[(index)]);
	dofVector2[Mesh.template getCellNextToCell<0,1>(index)]=fabsMin(-abs(a*1+b*0+c)*s,dofVector2[Mesh.template getCellNextToCell<0,1>(index)]);
	dofVector2[Mesh.template getCellNextToCell<1,1>(index)]=fabsMin(abs(a*1+b*1+c)*s,dofVector2[Mesh.template getCellNextToCell<1,1>(index)]);
	dofVector2[Mesh.template getCellNextToCell<1,0>(index)]=fabsMin(abs(a*0+b*1+c)*s,dofVector2[Mesh.template getCellNextToCell<1,0>(index)]);

}

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
void tnlFastSweeping< tnlGrid< 3,MeshReal, Device, MeshIndex >, Real, Index > :: setupSquare0110( Index i, Index j)
{
	Index index = Mesh.getCellIndex(CoordinatesType(i,j));
	dofVector2[index]=fabsMin(dofVector[index],dofVector2[(index)]);
	dofVector2[Mesh.template getCellNextToCell<0,1>(index)]=fabsMin(dofVector[Mesh.template getCellNextToCell<0,1>(index)],dofVector2[Mesh.template getCellNextToCell<0,1>(index)]);
	dofVector2[Mesh.template getCellNextToCell<1,1>(index)]=fabsMin(dofVector[Mesh.template getCellNextToCell<1,1>(index)],dofVector2[Mesh.template getCellNextToCell<1,1>(index)]);
	dofVector2[Mesh.template getCellNextToCell<1,0>(index)]=fabsMin(dofVector[Mesh.template getCellNextToCell<1,0>(index)],dofVector2[Mesh.template getCellNextToCell<1,0>(index)]);
}




#endif /* TNLFASTSWEEPING_IMPL_H_ */
