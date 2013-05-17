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

template< typename Real,
          typename Device,
          typename Index,
          template< int, typename, typename, typename > class GridGeometry >
tnlCentralFDMGradient< tnlGrid< 2, Real, Device, Index, GridGeometry > > :: tnlCentralFDMGradient()
: mesh( 0 )
{
}

template< typename Real,
          typename Device,
          typename Index,
          template< int, typename, typename, typename > class GridGeometry >
void tnlCentralFDMGradient< tnlGrid< 2, Real, Device, Index, GridGeometry > > :: bindMesh( const MeshType& mesh )
{
   this -> mesh = &mesh;
}

template< typename Real,
          typename Device,
          typename Index,
          template< int, typename, typename, typename > class GridGeometry >
   template< typename Vector >
void tnlCentralFDMGradient< tnlGrid< 2, Real, Device, Index, GridGeometry > > :: setFunction( Vector& f )
{
   this -> f. bind( f );
   this -> f. setName( tnlString( "bind Of " ) + f. getName() );
}

template< typename Real,
          typename Device,
          typename Index,
          template< int, typename, typename, typename > class GridGeometry >
void tnlCentralFDMGradient< tnlGrid< 2, Real, Device, Index, GridGeometry > > :: getGradient( const Index& c,
                                                                                              VertexType& grad_f  ) const
{
   tnlAssert( this -> mesh, cerr << "No mesh was set in tnlCentralFDMGradient. Use the bindMesh method." );

   const Index e = mesh -> getElementNeighbour( c,  1,  0 );
   const Index w = mesh -> getElementNeighbour( c, -1,  0 );
   const Index n = mesh -> getElementNeighbour( c,  0,  1 );
   const Index s = mesh -> getElementNeighbour( c,  0, -1 );
   CoordinatesType cCoordinates;
   mesh -> getElementCoordinates( c, cCoordinates );
   CoordinatesType eCoordinates( cCoordinates ),
                   wCoordinates( cCoordinates ),
                   nCoordinates( cCoordinates ),
                   sCoordinates( cCoordinates );
   eCoordinates. x() ++;
   wCoordinates. x() --;
   nCoordinates. y() ++;
   sCoordinates. y() --;


   VertexType eNormal, nNormal, wNormal, sNormal;
   mesh -> template getEdgeNormal<  1,  0 >( cCoordinates, eNormal );
   mesh -> template getEdgeNormal<  0,  1 >( cCoordinates, nNormal );
   mesh -> template getEdgeNormal< -1,  0 >( cCoordinates, wNormal );
   mesh -> template getEdgeNormal<  0, -1 >( cCoordinates, sNormal );
   const RealType cMu = mesh -> getElementMeasure( cCoordinates );
   const RealType eMu = mesh -> getElementMeasure( eCoordinates );
   const RealType nMu = mesh -> getElementMeasure( nCoordinates );
   const RealType wMu = mesh -> getElementMeasure( wCoordinates );
   const RealType sMu = mesh -> getElementMeasure( sCoordinates );
   const RealType f_e = ( cMu * f[ c ] + eMu * f[ e ] ) / ( cMu + eMu );
   const RealType f_n = ( cMu * f[ c ] + nMu * f[ n ] ) / ( cMu + nMu );
   const RealType f_w = ( cMu * f[ c ] + wMu * f[ w ] ) / ( cMu + wMu );
   const RealType f_s = ( cMu * f[ c ] + sMu * f[ s ] ) / ( cMu + sMu );
   grad_f. x() = 1.0 / cMu * ( f_e * eNormal. x() + f_n * nNormal. x() + f_w * wNormal. x() + f_s * sNormal. x() );
   grad_f. y() = 1.0 / cMu * ( f_e * eNormal. y() + f_n * nNormal. y() + f_w * wNormal. y() + f_s * sNormal. y() );


   //grad_f. x() = ( f[ e ] - f[ w ] ) / ( mesh -> getElementsDistance( eCoordinates, wCoordinates ) );
   //grad_f. y() = ( f[ n ] - f[ s ] ) / ( mesh -> getElementsDistance( nCoordinates, sCoordinates ) );
}

/****
 * Specialization for the grids with no deformations (Identical grid geometry)
 */

template< typename Real, typename Device, typename Index >
tnlCentralFDMGradient< tnlGrid< 2, Real, Device, Index, tnlIdenticalGridGeometry > > :: tnlCentralFDMGradient()
: mesh( 0 )
{
}

template< typename Real, typename Device, typename Index >
void tnlCentralFDMGradient< tnlGrid< 2, Real, Device, Index, tnlIdenticalGridGeometry > > :: bindMesh( const MeshType& mesh )
{
   this -> mesh = &mesh;
}

template< typename Real, typename Device, typename Index >
   template< typename Vector >
void tnlCentralFDMGradient< tnlGrid< 2, Real, Device, Index, tnlIdenticalGridGeometry > > :: setFunction( Vector& f )
{
   this -> f. bind( f );
   this -> f. setName( tnlString( "bind Of " ) + f. getName() );
}

template< typename Real, typename Device, typename Index >
void tnlCentralFDMGradient< tnlGrid< 2, Real, Device, Index, tnlIdenticalGridGeometry > > :: getGradient( const Index& i,
                                                                                                          VertexType& grad_f  ) const
{
   tnlAssert( this -> mesh, cerr << "No mesh was set in tnlCentralFDMGradient. Use the bindMesh method." );

   const Index e = mesh -> getElementNeighbour( i,  1,  0 );
   const Index w = mesh -> getElementNeighbour( i, -1,  0 );
   const Index n = mesh -> getElementNeighbour( i,  0,  1 );
   const Index s = mesh -> getElementNeighbour( i,  0, -1 );

   grad_f. x() = ( f[ e ] - f[ w ] ) / ( 2.0 * mesh -> getParametricStep(). x() );
   grad_f. y() = ( f[ n ] - f[ s ] ) / ( 2.0 * mesh -> getParametricStep(). y() );
}


#endif
