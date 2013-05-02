/***************************************************************************
                          tnlLinearDiffusion_impl.h  -  description
                             -------------------
    begin                : Apr 26, 2013
    copyright            : (C) 2013 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/

#ifndef TNLLINEARDIFFUSION_IMPL_H_
#define TNLLINEARDIFFUSION_IMPL_H_

template< typename Real, typename Device, typename Index >
tnlLinearDiffusion< tnlGrid< 2, Real, Device, Index > > :: tnlLinearDiffusion()
: mesh( 0 )
{
}

template< typename Real, typename Device, typename Index >
void tnlLinearDiffusion< tnlGrid< 2, Real, Device, Index > > :: bindMesh( const tnlGrid< 2, RealType, DeviceType, IndexType >& mesh )
{
   this -> mesh = &mesh;
}

template< typename Real, typename Device, typename Index >
   template< typename Vector >
void tnlLinearDiffusion< tnlGrid< 2, Real, Device, Index > > :: setFunction( Vector& f )
{
   this -> f. bind( f );
   this -> f. setName( tnlString( "bind Of " ) + f. getName() );
}

template< typename Real, typename Device, typename Index >
Real tnlLinearDiffusion< tnlGrid< 2, Real, Device, Index > > :: getDiffusion( const Index& c ) const
{
   tnlAssert( this -> mesh, cerr << "No mesh was set in tnlLinearDiffusion. Use the bindMesh method." );

   const Real hx = mesh -> getParametricStep(). x();
   const Real hy = mesh -> getParametricStep(). y();

   const Index e = mesh -> getElementNeighbour( c,  0,  1 );
   const Index w = mesh -> getElementNeighbour( c,  0, -1 );
   const Index n = mesh -> getElementNeighbour( c,  1,  0 );
   const Index s = mesh -> getElementNeighbour( c, -1,  0 );

   return ( f[ e ] - 2.0 * f[ c ] + f[ w ] ) / ( hx * hx ) +
          ( f[ n ] - 2.0 * f[ c ] + f[ s ] ) / ( hy * hy );
}

#endif
