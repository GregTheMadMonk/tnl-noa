/***************************************************************************
                          upwindEikonal1D_impl.h  -  description
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

#pragma once

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
Real upwindEikonalScheme< tnlGrid< 1,MeshReal, Device, MeshIndex >, Real, Index > :: positivePart(const Real arg) const
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
Real  upwindEikonalScheme< tnlGrid< 1,MeshReal, Device, MeshIndex >, Real, Index > :: negativePart(const Real arg) const
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
Real upwindEikonalScheme< tnlGrid< 1,MeshReal, Device, MeshIndex >, Real, Index > :: sign(const Real x, const Real eps) const
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
          typename Index >
bool upwindEikonalScheme< tnlGrid< 1,MeshReal, Device, MeshIndex >, Real, Index > :: init( const tnlParameterContainer& parameters )
{

	   const tnlString& meshFile = parameters.getParameter< tnlString >( "mesh" );
	   if( ! this->originalMesh.load( meshFile ) )
	   {
		   cerr << "I am not able to load the mesh from the file " << meshFile << "." << endl;
		   return false;
	   }


	   h = originalMesh.getSpaceSteps().x();

	   epsilon = parameters. getParameter< double >( "epsilon" );

	   if(epsilon != 0.0)
		   epsilon *=h;

	   /*dofVector. setSize( this->mesh.getDofs() );*/

	   return true;
}

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
tnlString upwindEikonalScheme< tnlGrid< 1, MeshReal, Device, MeshIndex >, Real, Index > :: getType()
{
   return tnlString( "upwindEikonalScheme< " ) +
          MeshType::getType() + ", " +
          ::getType< Real >() + ", " +
          ::getType< Index >() + " >";
}

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
template< typename PreimageFunction,
          typename MeshEntity >
__cuda_callable__
inline
Real
upwindEikonalScheme< tnlGrid< 1, MeshReal, Device, MeshIndex >, Real, Index >::
operator()( const PreimageFunction& u,
            const MeshEntity& entity,
            const Real& time ) const
{
	RealType nabla, xb, xf, signui;
	signui = sign( u( entity ), epsilon );
   
   const typename MeshEntity::template NeighbourEntities< 1 >& neighbourEntities = entity.getNeighbourEntities();

   const RealType& u_c = u[ entity.getIndex() ];
   const RealType& u_e = u[ neighbourEntities.template getEntityIndex< 1 >() ];
   const RealType& u_w = u[ neighbourEntities.template getEntityIndex< -1 >() ];
   if( signui > 0.0 )
   {
      xf = negativePart( ( u_e - u_c ) / h );
      xb = positivePart( ( u_c - u_w ) / h );
      nabla = sqrt( xf * xf + xb * xb );
      return signui * ( 1.0 - nabla );
   }
   else if (signui < 0.0)
   {
      xf = negativePart( ( u_w - u_c ) / h );
      xb = positivePart( ( u_c - u_e ) / h );
      nabla = sqrt( xf * xf + xb * xb );
      return signui * ( 1.0 - nabla );
   }
   else
   {
      return 0.0;
   }
}


