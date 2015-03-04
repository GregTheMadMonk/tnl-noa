/***************************************************************************
                          parallelGodunovEikonal2D_impl.h  -  description
                             -------------------
    begin                : Dec 1 , 2014
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

#ifndef GODUNOVEIKONAL2D_IMPL_H_
#define GODUNOVEIKONAL2D_IMPL_H_


template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
Real parallelGodunovEikonalScheme< tnlGrid< 2,MeshReal, Device, MeshIndex >, Real, Index >:: positivePart(const Real arg) const
{
	if(arg > 0.0)
		return arg;
	return 0.0;
}


template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
Real  parallelGodunovEikonalScheme< tnlGrid< 2,MeshReal, Device, MeshIndex >, Real, Index > :: negativePart(const Real arg) const
{
	if(arg < 0.0)
		return arg;
	return 0.0;
}


template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
Real parallelGodunovEikonalScheme< tnlGrid< 2,MeshReal, Device, MeshIndex >, Real, Index > :: sign(const Real x, const Real eps) const
{
	if(x > eps)
		return 1.0;
	else if (x < -eps)
		return (-1.0);

	if ( eps == 0.0)
		return 0.0;

	return sin((M_PI*x)/(2.0*eps));
}





template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
bool parallelGodunovEikonalScheme< tnlGrid< 2,MeshReal, Device, MeshIndex >, Real, Index > :: init( const tnlParameterContainer& parameters )
{

	   const tnlString& meshFile = parameters.GetParameter< tnlString >( "mesh" );
	   MeshType originalMesh;
	   if( ! originalMesh.load( meshFile ) )
	   {
		   cerr << "I am not able to load the mesh from the file " << meshFile << "." << endl;
		   return false;
	   }




	   hx = originalMesh.getHx();
	   hy = originalMesh.getHy();

	   epsilon = parameters. GetParameter< double >( "epsilon" );

	   epsilon *=sqrt( hx*hx + hy*hy );

//	   dofVector. setSize( this->mesh.getDofs() );

	   return true;

}


template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
tnlString parallelGodunovEikonalScheme< tnlGrid< 2, MeshReal, Device, MeshIndex >, Real, Index > :: getType()
{
   return tnlString( "tnlLinearDiffusion< " ) +
          MeshType::getType() + ", " +
          ::getType< Real >() + ", " +
          ::getType< Index >() + " >";
}


template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
template< typename Vector >
#ifdef HAVE_CUDA
__device__ __host__
#endif
Real parallelGodunovEikonalScheme< tnlGrid< 2, MeshReal, Device, MeshIndex >, Real, Index >:: getValue( const MeshType& mesh,
          	  	  	  	  	  	  	  	  	  	  	  	  	  	  	  	  	  	  	  	 const IndexType cellIndex,
          	  	  	  	  	  	  	  	  	  	  	  	  	  	  	  	  	  	  	  	 const CoordinatesType& coordinates,
          	  	  	  	  	  	  	  	  	  	  	  	  	  	  	  	  	  	  	  	 const Vector& u,
          	  	  	  	  	  	  	  	  	  	  	  	  	  	  	  	  	  	  	  	 const Real& time,
          	  	  	  	  	  	  	  	  	  	  	  	  	  	  	  	  	  	  	  	 const IndexType boundaryCondition ) const
{

	if ( ((coordinates.x() == 0 && (boundaryCondition & 4)) or
		 (coordinates.x() == mesh.getDimensions().x() - 1 && (boundaryCondition & 2)) or
		 (coordinates.y() == 0 && (boundaryCondition & 8)) or
		 (coordinates.y() == mesh.getDimensions().y() - 1  && (boundaryCondition & 1)))
		 /*and
		 !(		 (coordinates.y() == 0 or coordinates.y() == mesh.getDimensions().y() - 1)
				 and
				 ( coordinates.x() == 0 or coordinates.x() == mesh.getDimensions().x() - 1)
		  )*/
		)
	{
		return 0.0;
	}

	RealType acc = hx*hy*hx*hy;

	RealType nabla, xb, xf, yb, yf, signui;

	signui = sign(u[cellIndex],epsilon);

	if(fabs(u[cellIndex]) < acc) return 0.0;

	   if(signui > 0.0)
	   {
	/**/ /*  if(boundaryCondition & 2)
			   xf = (u[mesh.getCellXSuccessor( cellIndex )] - u[cellIndex])/hx;
		   else *//*if(boundaryCondition & 4)
			   xf = 0.0;
		   else /**/if(coordinates.x() == mesh.getDimensions().x() - 1)
			   xf = negativePart((u[mesh.getCellXPredecessor( cellIndex )] - u[cellIndex])/hx);
		   else
			   xf = negativePart((u[mesh.getCellXSuccessor( cellIndex )] - u[cellIndex])/hx);

	/**/ /*  if(boundaryCondition & 4)
			   xb = (u[cellIndex] - u[mesh.getCellXPredecessor( cellIndex )])/hx;
		   else *//*if(boundaryCondition & 2)
			   xb = 0.0;
		   else /**/if(coordinates.x() == 0)
			   xb = positivePart((u[cellIndex] - u[mesh.getCellXSuccessor( cellIndex )])/hx);
		   else
			   xb = positivePart((u[cellIndex] - u[mesh.getCellXPredecessor( cellIndex )])/hx);

	/**/  /* if(boundaryCondition & 1)
			   yf = (u[mesh.getCellYSuccessor( cellIndex )] - u[cellIndex])/hy;
		   else *//*if(boundaryCondition & 8)
			   yf = 0.0;
		   else /**/if(coordinates.y() == mesh.getDimensions().y() - 1)
			   yf = negativePart((u[mesh.getCellYPredecessor( cellIndex )] - u[cellIndex])/hy);
		   else
			   yf = negativePart((u[mesh.getCellYSuccessor( cellIndex )] - u[cellIndex])/hy);

	/**/  /* if(boundaryCondition & 8)
			   yb = (u[cellIndex] - u[mesh.getCellYPredecessor( cellIndex )])/hy;
		   else *//*if(boundaryCondition & 1)
			   yb = 0.0;
		   else /**/if(coordinates.y() == 0)
			   yb = positivePart((u[cellIndex] - u[mesh.getCellYSuccessor( cellIndex )])/hy);
		   else
			   yb = positivePart((u[cellIndex] - u[mesh.getCellYPredecessor( cellIndex )])/hy);

		   if(xb + xf > 0.0)
			   xf = 0.0;
		   else
			   xb = 0.0;

		   if(yb + yf > 0.0)
			   yf = 0.0;
		   else
			   yb = 0.0;

		   nabla = sqrt (xf*xf + xb*xb + yf*yf + yb*yb );
		   if(fabs(1.0-nabla) < acc)
			   return 0.0;
		   return signui*(1.0 - nabla);
	   }
	   else if (signui < 0.0)
	   {

	/**/  /* if(boundaryCondition & 2)
			   xf = (u[mesh.getCellXSuccessor( cellIndex )] - u[cellIndex])/hx;
		   else*//* if(boundaryCondition & 4)
			   xf = 0.0;
		   else /**/if(coordinates.x() == mesh.getDimensions().x() - 1)
			   xf = positivePart((u[mesh.getCellXPredecessor( cellIndex )] - u[cellIndex])/hx);
		   else
			   xf = positivePart((u[mesh.getCellXSuccessor( cellIndex )] - u[cellIndex])/hx);

	/**/  /* if(boundaryCondition & 4)
			   xb = (u[cellIndex] - u[mesh.getCellXPredecessor( cellIndex )])/hx;
		   else*//* if(boundaryCondition & 2)
			   xb = 0.0;
		   else /**/if(coordinates.x() == 0)
			   xb = negativePart((u[cellIndex] - u[mesh.getCellXSuccessor( cellIndex )])/hx);
		   else
			   xb = negativePart((u[cellIndex] - u[mesh.getCellXPredecessor( cellIndex )])/hx);

	/**/ /*  if(boundaryCondition & 1)
			   yf = (u[mesh.getCellYSuccessor( cellIndex )] - u[cellIndex])/hy;
		   else *//*if(boundaryCondition & 8)
			   yf = 0.0;
		   else /**/if(coordinates.y() == mesh.getDimensions().y() - 1)
			   yf = positivePart((u[mesh.getCellYPredecessor( cellIndex )] - u[cellIndex])/hy);
		   else
			   yf = positivePart((u[mesh.getCellYSuccessor( cellIndex )] - u[cellIndex])/hy);

	/**/  /* if(boundaryCondition & 8)
			   yb = (u[cellIndex] - u[mesh.getCellYPredecessor( cellIndex )])/hy;
		   else*//* if(boundaryCondition & 1)
			   yb = 0.0;
		   else /**/if(coordinates.y() == 0)
			   yb = negativePart((u[cellIndex] - u[mesh.getCellYSuccessor( cellIndex )])/hy);
		   else
			   yb = negativePart((u[cellIndex] - u[mesh.getCellYPredecessor( cellIndex )])/hy);


		   if(xb + xf > 0.0)
			   xb = 0.0;
		   else
			   xf = 0.0;

		   if(yb + yf > 0.0)
			   yb = 0.0;
		   else
			   yf = 0.0;

		   nabla = sqrt (xf*xf + xb*xb + yf*yf + yb*yb );

		   if(fabs(1.0-nabla) < acc)
			   return 0.0;
		   return signui*(1.0 - nabla);
	   }
	   else
	   {
		   return 0.0;
	   }

}


#endif /* GODUNOVEIKONAL2D_IMPL_H_ */
