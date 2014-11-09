/***************************************************************************
                          upwind1D_impl.h  -  description
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

#ifndef UPWIND1D_IMPL_H_
#define UPWIND1D_IMPL_H_



template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index,
		  typename Function >
Real upwindScheme< tnlGrid< 1,MeshReal, Device, MeshIndex >, Real, Index, Function > :: positivePart(const Real arg) const
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
Real  upwindScheme< tnlGrid< 1,MeshReal, Device, MeshIndex >, Real, Index, Function > :: negativePart(const Real arg) const
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
Real upwindScheme< tnlGrid< 1,MeshReal, Device, MeshIndex >, Real, Index, Function > :: sign(const Real x, const Real eps) const
{
	if(x > eps)
		return 1.0;
	else if (x < -eps)
		return (-1.0);

	if(eps == 0.0)
		return 0.0;

	return sin( ( M_PI * x ) / (2 * eps) );
}





template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index,
		  typename Function >
bool upwindScheme< tnlGrid< 1,MeshReal, Device, MeshIndex >, Real, Index, Function > :: init( const tnlParameterContainer& parameters )
{

	   const tnlString& meshFile = parameters.GetParameter< tnlString >( "mesh" );
	   if( ! this->originalMesh.load( meshFile ) )
	   {
		   cerr << "I am not able to load the mesh from the file " << meshFile << "." << endl;
		   return false;
	   }


	   h = originalMesh.getHx();

	   epsilon = parameters. GetParameter< double >( "epsilon" );

	   if(epsilon != 0.0)
		   epsilon *=h;

	   f.setup( parameters );

	   /*dofVector. setSize( this->mesh.getDofs() );*/

	   return true;
}

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index,
		  typename Function >
tnlString upwindScheme< tnlGrid< 1, MeshReal, Device, MeshIndex >, Real, Index, Function > :: getType()
{
   return tnlString( "upwindScheme< " ) +
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
Real upwindScheme< tnlGrid< 1, MeshReal, Device, MeshIndex >, Real, Index, Function >:: getValue( const MeshType& mesh,
          	  	  	  	  	  	  	  	  	  	  	  	  	  	  	  	  	  	  	  	 const IndexType cellIndex,
          	  	  	  	  	  	  	  	  	  	  	  	  	  	  	  	  	  	  	  	 const CoordinatesType& coordinates,
          	  	  	  	  	  	  	  	  	  	  	  	  	  	  	  	  	  	  	  	 const Vector& u,
          	  	  	  	  	  	  	  	  	  	  	  	  	  	  	  	  	  	  	  	 const Real& time ) const
{

	RealType nabla, xb, xf, fi;

	fi = f.getValue(coordinates,0);

	   if(fi > 0.0)
	   {
		   xf = negativePart((u[mesh.getCellXSuccessor( cellIndex )] - u[cellIndex])/h);
		   xb = positivePart((u[cellIndex] - u[mesh.getCellXPredecessor( cellIndex )])/h);

		   nabla = sqrt (xf*xf + xb*xb );

		   return -fi*( nabla);
	   }
	   else if (fi < 0.0)
	   {
		   xf = positivePart((u[mesh.getCellXSuccessor( cellIndex )] - u[cellIndex])/h);
		   xb = negativePart((u[cellIndex] - u[mesh.getCellXPredecessor( cellIndex )])/h);

		   nabla = sqrt (xf*xf + xb*xb );

		   return -fi*( nabla);
	   }
	   else
	   {
		   return 0.0;
	   }

}


#endif /* UPWIND1D_IMPL_H_ */
