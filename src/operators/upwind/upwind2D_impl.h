/***************************************************************************
                          upwind2D_impl.h  -  description
                             -------------------
    begin                : Jul 8 , 2014
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

#ifndef UPWIND2D_IMPL_H_
#define UPWIND2D_IMPL_H_


template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index,
		  typename Function >
Real upwindScheme< tnlGrid< 2,MeshReal, Device, MeshIndex >, Real, Index, Function >:: positivePart(const Real arg) const
{
	if(arg > 0.0)
		return arg;
	return 0.0;
}


template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index,
		  typename Function >
Real  upwindScheme< tnlGrid< 2,MeshReal, Device, MeshIndex >, Real, Index, Function > :: negativePart(const Real arg) const
{
	if(arg < 0.0)
		return arg;
	return 0.0;
}


template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index,
		  typename Function >
Real upwindScheme< tnlGrid< 2,MeshReal, Device, MeshIndex >, Real, Index, Function > :: sign(const Real x, const Real eps) const
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
          typename Index,
		  typename Function >
bool upwindScheme< tnlGrid< 2,MeshReal, Device, MeshIndex >, Real, Index, Function > :: init( const tnlParameterContainer& parameters )
{

	   const tnlString& meshFile = parameters.getParameter< tnlString >( "mesh" );
	   if( ! this->originalMesh.load( meshFile ) )
	   {
		   cerr << "I am not able to load the mesh from the file " << meshFile << "." << endl;
		   return false;
	   }

	   hx = originalMesh.getHx();
	   hy = originalMesh.getHy();

	   epsilon = parameters. getParameter< double >( "epsilon" );

	   if(epsilon != 0.0)
		   epsilon *=sqrt( hx*hx + hy*hy );

	   f.setup( parameters );

//	   dofVector. setSize( this->mesh.getDofs() );

	   return true;

}


template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index,
		  typename Function >
tnlString upwindScheme< tnlGrid< 2, MeshReal, Device, MeshIndex >, Real, Index, Function > :: getType()
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
          typename Index,
		  typename Function >
template< typename Vector >
#ifdef HAVE_CUDA
__device__ __host__
#endif
Real upwindScheme< tnlGrid< 2, MeshReal, Device, MeshIndex >, Real, Index, Function >:: getValue( const MeshType& mesh,
          	  	  	  	  	  	  	  	  	  	  	  	  	  	  	  	  	  	  	  	 const IndexType cellIndex,
          	  	  	  	  	  	  	  	  	  	  	  	  	  	  	  	  	  	  	  	 const CoordinatesType& coordinates,
          	  	  	  	  	  	  	  	  	  	  	  	  	  	  	  	  	  	  	  	 const Vector& u,
          	  	  	  	  	  	  	  	  	  	  	  	  	  	  	  	  	  	  	  	 const Real& time ) const
{

	RealType nabla, xb, xf, yb, yf, fi;

	fi = f.getValue(mesh.getCellCenter( coordinates ),time);

	   if(fi > 0.0)
	   {
		   xf = negativePart((u[mesh.getCellXSuccessor( cellIndex )] - u[cellIndex])/hx);
		   xb = positivePart((u[cellIndex] - u[mesh.getCellXPredecessor( cellIndex )])/hx);
		   yf = negativePart((u[mesh.getCellYSuccessor( cellIndex )] - u[cellIndex])/hy);
		   yb = positivePart((u[cellIndex] - u[mesh.getCellYPredecessor( cellIndex )])/hy);

		   nabla = sqrt (xf*xf + xb*xb + yf*yf + yb*yb );

		   return -fi*( nabla);
	   }
	   else if (fi < 0.0)
	   {
		   xf = positivePart((u[mesh.getCellXSuccessor( cellIndex )] - u[cellIndex])/hx);
		   xb = negativePart((u[cellIndex] - u[mesh.getCellXPredecessor( cellIndex )])/hx);
		   yf = positivePart((u[mesh.getCellYSuccessor( cellIndex )] - u[cellIndex])/hy);
		   yb = negativePart((u[cellIndex] - u[mesh.getCellYPredecessor( cellIndex )])/hy);

		   nabla = sqrt (xf*xf + xb*xb + yf*yf + yb*yb );

		   return -fi*( nabla);
	   }
	   else
	   {
		   return 0.0;
	   }

}

#endif /* UPWIND2D_IMPL_H_ */
