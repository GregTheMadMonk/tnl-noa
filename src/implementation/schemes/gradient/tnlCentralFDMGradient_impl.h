/***************************************************************************
                          tnlCentralFDMGradient_impl.h  -  description
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

#ifndef TNLCENTRALFDMGRADIENT_IMPL_H_
#define TNLCENTRALFDMGRADIENT_IMPL_H_

template< typename Real, typename Device, typename Index >
tnlCentralFDMGradient< tnlGrid< 2, Real, Device, Index > > :: tnlCentralFDMGradient()
: mesh( 0 )
{
}

template< typename Real, typename Device, typename Index >
void tnlCentralFDMGradient< tnlGrid< 2, Real, Device, Index > > :: bindMesh( const tnlGrid< 2, RealType, DeviceType, IndexType >& mesh )
{
   this -> mesh = mesh;
}

template< typename Real, typename Device, typename Index >
   template< typename Vector >
void tnlCentralFDMGradient< tnlGrid< 2, Real, Device, Index > > :: setFunction( Vector& f )
{
   this -> f. bind( f );
}

template< typename Real, typename Device, typename Index >
void tnlCentralFDMGradient< tnlGrid< 2, Real, Device, Index > > :: getGradient( const Index& i,
                                                                     RealType& f_x,
                                                                     RealType& f_y ) const
{
   tnlAssert( this -> mesh, cerr << "No mesh was set in tnlCentralFDMGradient" );

   const Index e = mesh -> getElementNeighbour( i,  1,  0 );
   const Index w = mesh -> getElementNeighbour( i, -1,  0 );
   const Index n = mesh -> getElementNeighbour( i,  0,  1 );
   const Index s = mesh -> getElementNeighbour( i,  0, -1 );

   f_x = ( f[ e ] - f[ w ] ) / ( 2.0 * mesh -> getSpaceStep(). x() );
   f_y = ( f[ n ] - f[ s ] ) / ( 2.0 * mesh -> getSpaceStep(). y() );
}

#endif
