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

   const Index e = mesh -> getElementNeighbour( c,  1,  0 );
   const Index w = mesh -> getElementNeighbour( c, -1,  0 );
   const Index n = mesh -> getElementNeighbour( c,  0,  1 );
   const Index s = mesh -> getElementNeighbour( c,  0, -1 );

   //return ( f[ e ] - 2.0 * f[ c ] + f[ w ] ) / ( hx * hx ) +
   //       ( f[ n ] - 2.0 * f[ c ] + f[ s ] ) / ( hy * hy );

   /****
    * Diffusion for the deformed grids
    */
   const Index en = mesh -> getElementNeighbour( c,  1,  1 );
   const Index es = mesh -> getElementNeighbour( c,  1, -1 );
   const Index wn = mesh -> getElementNeighbour( c, -1,  1 );
   const Index ws = mesh -> getElementNeighbour( c, -1, -1 );

   CoordinatesType cCoordinates, eCoordinates, nCoordinates, wCoordinates, sCoordinates,
                  enCoordinates, esCoordinates, wnCoordinates, wsCoordinates;
   this -> mesh -> getElementCoordinates( c, cCoordinates );
   eCoordinates = nCoordinates = wCoordinates = sCoordinates = cCoordinates;
   eCoordinates. x() ++;
   wCoordinates. x() --;
   nCoordinates. y() ++;
   sCoordinates. y() --;
   enCoordinates = esCoordinates = eCoordinates;
   enCoordinates. y() ++;
   esCoordinates. y() --;
   wnCoordinates = wsCoordinates = wCoordinates;
   wnCoordinates. y() ++;
   wsCoordinates. y() --;

   const RealType cMeasure =  this -> mesh -> getElementMeasure( cCoordinates );
   const RealType eMeasure =  this -> mesh -> getElementMeasure( eCoordinates );
   const RealType wMeasure =  this -> mesh -> getElementMeasure( wCoordinates );
   const RealType nMeasure =  this -> mesh -> getElementMeasure( nCoordinates );
   const RealType sMeasure =  this -> mesh -> getElementMeasure( sCoordinates );
   const RealType enMeasure =  this -> mesh -> getElementMeasure( enCoordinates );
   const RealType esMeasure =  this -> mesh -> getElementMeasure( esCoordinates );
   const RealType wnMeasure =  this -> mesh -> getElementMeasure( wnCoordinates );
   const RealType wsMeasure =  this -> mesh -> getElementMeasure( wsCoordinates );

   const RealType f_en = ( cMeasure * f[ c ] + eMeasure * f[ e ] +
                           nMeasure * f[ n ] + enMeasure * f[ en ] ) /
                         ( cMeasure + eMeasure + nMeasure + enMeasure );
   const RealType f_es = ( cMeasure * f[ c ] + eMeasure * f[ e ] +
                           sMeasure * f[ s ] + esMeasure * f[ es ] ) /
                         ( cMeasure + eMeasure + sMeasure + esMeasure );
   const RealType f_wn = ( cMeasure * f[ c ] + wMeasure * f[ w ] +
                           nMeasure * f[ n ] + wnMeasure * f[ wn ] ) /
                         ( cMeasure + wMeasure + nMeasure + wnMeasure );
   const RealType f_ws = ( cMeasure * f[ c ] + wMeasure * f[ w ] +
                           sMeasure * f[ s ] + wsMeasure * f[ ws ] ) /
                         ( cMeasure + wMeasure + sMeasure + wsMeasure );

   const RealType f_cen = 0.5 * ( f[ c ] + f_en );
   const RealType f_ces = 0.5 * ( f[ c ] + f_es );
   const RealType f_cwn = 0.5 * ( f[ c ] + f_wn );
   const RealType f_cws = 0.5 * ( f[ c ] + f_ws );

   VertexType cCenter, eCenter, nCenter, wCenter, sCenter;
   this -> mesh -> getElementCenter( cCoordinates, cCenter );
   this -> mesh -> getElementCenter( eCoordinates, eCenter );
   this -> mesh -> getElementCenter( wCoordinates, wCenter );
   this -> mesh -> getElementCenter( nCoordinates, nCenter );
   this -> mesh -> getElementCenter( sCoordinates, sCenter );

   VertexType enVertex, esVertex, wnVertex, wsVertex;
   this -> mesh -> template getVertex<  1,  1 >( cCoordinates, enVertex );
   this -> mesh -> template getVertex<  1, -1 >( cCoordinates, esVertex );
   this -> mesh -> template getVertex< -1,  1 >( cCoordinates, wnVertex );
   this -> mesh -> template getVertex< -1, -1 >( cCoordinates, wsVertex );

   const RealType f_x_e = 1.0 / this -> mesh -> template getElementCoVolumeMeasure< 1, 0 >( cCoordinates ) *
                          ( f_cen * ( enVertex. y() - cCenter. y() ) +
                            f_ces * ( esVertex. y() - cCenter. y() ) +
                            0.5 * ( f_es + f[ e ] ) * ( esVertex. y() - eCenter. y() ) +
                            0.5 * ( f_en + f[ e ] ) * ( enVertex. y() - eCenter. y() ) );
   const RealType f_y_e = 1.0 / this -> mesh -> template getElementCoVolumeMeasure< 1, 0 >( cCoordinates ) *
                          ( f_cen * ( enVertex. x() - cCenter. x() ) +
                            f_ces * ( esVertex. x() - cCenter. x() ) +
                            0.5 * ( f_es + f[ e ] ) * ( esVertex. x() - eCenter. x() ) +
                            0.5 * ( f_en + f[ e ] ) * ( enVertex. x() - eCenter. x() ) );
   const RealType f_x_w = 1.0 / this -> mesh -> template getElementCoVolumeMeasure< -1, 0 >( cCoordinates ) *
                          ( f_cwn * ( wnVertex. y() - cCenter. y() ) +
                            f_cws * ( wsVertex. y() - cCenter. y() ) +
                            0.5 * ( f_ws + f[ w ] ) * ( wsVertex. y() - wCenter. y() ) +
                            0.5 * ( f_wn + f[ w ] ) * ( wnVertex. y() - wCenter. y() ) );
   const RealType f_y_w = 1.0 / this -> mesh -> template getElementCoVolumeMeasure< -1, 0 >( cCoordinates ) *
                          ( f_cwn * ( wnVertex. x() - cCenter. x() ) +
                            f_cws * ( wsVertex. x() - cCenter. x() ) +
                            0.5 * ( f_ws + f[ w ] ) * ( wsVertex. x() - wCenter. x() ) +
                            0.5 * ( f_wn + f[ w ] ) * ( wnVertex. x() - wCenter. x() ) );
   const RealType f_x_n = 1.0 / this -> mesh -> template getElementCoVolumeMeasure< 0, 1 >( cCoordinates ) *
                          ( f_cen * ( enVertex. y() - cCenter. y() ) +
                            f_cwn * ( wnVertex. y() - cCenter. y() ) +
                            0.5 * ( f_en + f[ n ] ) * ( enVertex. y() - nCenter. y() ) +
                            0.5 * ( f_wn + f[ n ] ) * ( wnVertex. y() - nCenter. y() ) );
   const RealType f_y_n = 1.0 / this -> mesh -> template getElementCoVolumeMeasure< 0, 1 >( cCoordinates ) *
                          ( f_cen * ( enVertex. x() - cCenter. x() ) +
                            f_cwn * ( wnVertex. x() - cCenter. x() ) +
                            0.5 * ( f_en + f[ n ] ) * ( enVertex. x() - nCenter. x() ) +
                            0.5 * ( f_wn + f[ n ] ) * ( wnVertex. x() - nCenter. x() ) );
   const RealType f_x_s = 1.0 / this -> mesh -> template getElementCoVolumeMeasure< 0, -1 >( cCoordinates ) *
                          ( f_ces * ( esVertex. y() - cCenter. y() ) +
                            f_cws * ( wsVertex. y() - cCenter. y() ) +
                            0.5 * ( f_es + f[ s ] ) * ( esVertex. y() - sCenter. y() ) +
                            0.5 * ( f_ws + f[ s ] ) * ( wsVertex. y() - sCenter. y() ) );
   const RealType f_y_s = 1.0 / this -> mesh -> template getElementCoVolumeMeasure< 0, -1 >( cCoordinates ) *
                          ( f_ces * ( esVertex. x() - cCenter. x() ) +
                            f_cws * ( wsVertex. x() - cCenter. x() ) +
                            0.5 * ( f_es + f[ s ] ) * ( enVertex. x() - sCenter. x() ) +
                            0.5 * ( f_ws + f[ s ] ) * ( wnVertex. x() - sCenter. x() ) );

   VertexType eNormal, nNormal, wNormal, sNormal;
   this -> mesh -> template getEdgeNormal<  1,  0 >( cCoordinates, eNormal );
   this -> mesh -> template getEdgeNormal< -1,  0 >( cCoordinates, wNormal );
   this -> mesh -> template getEdgeNormal<  0,  1 >( cCoordinates, nNormal );
   this -> mesh -> template getEdgeNormal<  0, -1 >( cCoordinates, sNormal );
   return 1.0 / this -> mesh -> getElementMeasure( cCoordinates ) *
          ( f_x_e * eNormal. x() + f_x_w * wNormal. x() +
            f_y_n * nNormal. y() + f_y_s * sNormal. y() );
}

#endif
