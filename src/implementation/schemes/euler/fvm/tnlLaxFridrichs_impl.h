/***************************************************************************
                          tnlLaxFridrichs_impl.h  -  description
                             -------------------
    begin                : Mar 1, 2013
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

#ifndef TNLLAXFRIDRICHS_IMPL_H_
#define TNLLAXFRIDRICHS_IMPL_H_


template< typename Real,
          typename Device,
          typename Index,
          typename PressureGradient >
tnlLaxFridrichs< tnlGrid< 2, Real, Device, Index >,
                 PressureGradient > :: tnlLaxFridrichs()
: regularizeEps( 0.0 ),
  viscosityCoefficient( 1.0 ),
  mesh( 0 ),
  pressureGradient( 0 )
{
}

template< typename Real,
          typename Device,
          typename Index,
          typename PressureGradient >
void tnlLaxFridrichs< tnlGrid< 2, Real, Device, Index >, PressureGradient  > :: bindMesh( const MeshType& mesh )
{
   this -> mesh = &mesh;
}

template< typename Real,
          typename Device,
          typename Index,
          typename PressureGradient >
void tnlLaxFridrichs< tnlGrid< 2, Real, Device, Index >, PressureGradient  > :: setRegularization( const RealType& epsilon )
{
   this -> regularizeEps = epsilon;
}

template< typename Real,
          typename Device,
          typename Index,
          typename PressureGradient >
void tnlLaxFridrichs< tnlGrid< 2, Real, Device, Index >, PressureGradient  > :: setViscosityCoefficient( const RealType& v )
{
   this -> viscosityCoefficient = v;
}

template< typename Real,
          typename Device,
          typename Index,
          typename PressureGradient >
   template< typename Vector >
void tnlLaxFridrichs< tnlGrid< 2, Real, Device, Index >, PressureGradient  > :: setRho( Vector& rho )
{
   this -> rho. bind( rho );
   this -> rho. setName( tnlString( "bind Of " ) + rho. getName() );
}

template< typename Real,
          typename Device,
          typename Index,
          typename PressureGradient >
   template< typename Vector >
void tnlLaxFridrichs< tnlGrid< 2, Real, Device, Index >, PressureGradient  > :: setRhoU1( Vector& rho_u1 )
{
   this -> rho_u1. bind( rho_u1 );
   this -> rho_u1. setName( tnlString( "bind Of " ) + rho_u1. getName() );
}

template< typename Real,
          typename Device,
          typename Index,
          typename PressureGradient >
   template< typename Vector >
void tnlLaxFridrichs< tnlGrid< 2, Real, Device, Index >, PressureGradient  > :: setRhoU2( Vector& rho_u2 )
{
   this -> rho_u2. bind( rho_u2 );
   this -> rho_u2. setName( tnlString( "bind Of " ) + rho_u2. getName() );
}

template< typename Real,
          typename Device,
          typename Index,
          typename PressureGradient >
   template< typename Vector >
void tnlLaxFridrichs< tnlGrid< 2, Real, Device, Index >, PressureGradient  > :: setPressureGradient( Vector& grad_p )
{
   this -> pressureGradient = &grad_p;
}

template< typename Real,
          typename Device,
          typename Index,
          typename PressureGradient >
void tnlLaxFridrichs< tnlGrid< 2, Real, Device, Index >, PressureGradient  > :: getExplicitRhs( const IndexType centralVolume,
                                                                                                RealType& rho_t,
                                                                                                RealType& rho_u1_t,
                                                                                                RealType& rho_u2_t ) const
{
   tnlAssert( mesh, cerr << "No mesh has been binded with the Lax-Fridrichs scheme." );
   tnlAssert( pressureGradient, cerr << "No pressure gradient was set in the the Lax-Fridrichs scheme." )

   const IndexType& xSize = this -> mesh -> getDimensions(). x();
   const IndexType& ySize = this -> mesh -> getDimensions(). y();
   const RealType hx = this -> mesh -> getParametricStep(). x();
   const RealType hy = this -> mesh -> getParametricStep(). y();

   const IndexType& c = centralVolume;
   const IndexType e = this -> mesh -> getElementNeighbour( centralVolume,  0,  1 );
   const IndexType w = this -> mesh -> getElementNeighbour( centralVolume,  0, -1 );
   const IndexType n = this -> mesh -> getElementNeighbour( centralVolume,  1,  0 );
   const IndexType s = this -> mesh -> getElementNeighbour( centralVolume, -1,  0 );

   /****
    * rho_t + ( rho u_1 )_x + ( rho u_2 )_y =  0
    */
   const RealType u1_e = rho_u1[ e ] / regularize( rho[ e ] );
   const RealType u1_w = rho_u1[ w ] / regularize( rho[ w ] );
   const RealType u2_n = rho_u2[ n ] / regularize( rho[ n ] );
   const RealType u2_s = rho_u2[ s ] / regularize( rho[ s ] );
   rho_t= this -> viscosityCoefficient * 0.25 * ( rho[ e ] + rho[ w ] + rho[ s ] + rho[ n ] - 4.0 * rho[ c ] )
               - ( rho[ e ] * u1_e - rho[ w ] * u1_w ) / ( 2.0 * hx )
               - ( rho[ n ] * u2_n - rho[ s ] * u2_s ) / ( 2.0 * hy );

   /****
    * Compute the pressure gradient
    */
   RealType p_x( 0.0 ), p_y( 0.0 );
   pressureGradient -> getGradient( c, p_x, p_y );

   /****
    * ( rho * u1 )_t + ( rho * u1 * u1 )_x + ( rho * u1 * u2 )_y - p_x =  0
    */
   rho_u1_t = this -> viscosityCoefficient * 0.25 * ( rho_u1[ e ] + rho_u1[ w ] + rho_u1[ s ] + rho_u1[ n ] - 4.0 * rho_u1[ c ] )
                   - ( rho_u1[ e ] * u1_e - rho_u1[ w ] * u1_w ) / ( 2.0 * hx )
                   - ( rho_u1[ n ] * u2_n - rho_u1[ s ] * u2_s ) / ( 2.0 * hy )
                   - p_x;
   /****
    * ( rho * u1 )_t + ( rho * u1 * u1 )_x + ( rho * u1 * u2 )_y - p_y =  0
    */
   rho_u2_t = this -> viscosityCoefficient * 0.25 * ( rho_u2[ e ] + rho_u2[ w ] + rho_u2[ s ] + rho_u2[ n ] - 4.0 * rho_u2[ c ] )
                   - ( rho_u2[ e ] * u1_e - rho_u2[ w ] * u1_w ) / ( 2.0 * hx )
                   - ( rho_u2[ n ] * u2_n - rho_u2[ s ] * u2_s ) / ( 2.0 * hy )
                   - p_y;

   /****
    * Scheme for deformed grids
    */
   const RealType u1_c = rho_u1[ c ] / regularize( rho[ c ] );
   const RealType u1_n = rho_u1[ n ] / regularize( rho[ n ] );
   const RealType u1_s = rho_u1[ s ] / regularize( rho[ s ] );
   const RealType u2_c = rho_u2[ c ] / regularize( rho[ c ] );
   const RealType u2_e = rho_u2[ e ] / regularize( rho[ e ] );
   const RealType u2_w = rho_u2[ w ] / regularize( rho[ w ] );


   /****
    * Get the central volume and its neighbours (east, north, west, south) coordinates
    */
   tnlTuple< 2, IndexType > c_coordinates, e_coordinates, n_coordinates, w_coordinates, s_coordinates;
   this -> mesh -> getElementCoordinates( c, c_coordinates );
   e_coordinates = n_coordinates = w_coordinates = s_coordinates = c_coordinates;
   e_coordinates. x() ++;
   w_coordinates. x() --;
   n_coordinates. y() ++;
   s_coordinates. y() --;

   /****
    * Get the volumes measure
    */
   const RealType mu_D_c = this -> mesh -> getElementMeasure( c_coordinates );
   const RealType mu_D_e = this -> mesh -> getElementMeasure( e_coordinates );
   const RealType mu_D_n = this -> mesh -> getElementMeasure( n_coordinates );
   const RealType mu_D_w = this -> mesh -> getElementMeasure( w_coordinates );
   const RealType mu_D_s = this -> mesh -> getElementMeasure( s_coordinates );

   /****
    * Get the volumes centers of gravity
    */
   tnlTuple< 2, RealType > c_center, e_center, w_center, n_center, s_center;
   this -> mesh -> getElementCenter( c_coordinates, c_center );
   this -> mesh -> getElementCenter( c_coordinates, e_center );
   this -> mesh -> getElementCenter( c_coordinates, w_center );
   this -> mesh -> getElementCenter( c_coordinates, n_center );
   this -> mesh -> getElementCenter( c_coordinates, s_center );

   /****
    * Get delta x and delta y between the volumes
    */
   const RealType dx_e = e_center. x() - c_center. x();
   const RealType dx_w = w_center. x() - w_center. x();
   const RealType dx_n = n_center. x() - n_center. x();
   const RealType dx_s = s_center. x() - s_center. x();
   const RealType dy_e = e_center. y() - c_center. y();
   const RealType dy_w = w_center. y() - w_center. y();
   const RealType dy_n = n_center. y() - n_center. y();
   const RealType dy_s = s_center. y() - s_center. y();

   /****
    * Compute the fluxes
    */
   const RealType rho_f_e = 0.5 * ( rho[ c ] * u1_c + rho[ e ] * u1_e );
   const RealType rho_f_w = 0.5 * ( rho[ c ] * u1_c + rho[ w ] * u1_w );
   const RealType rho_f_n = 0.5 * ( rho[ c ] * u1_c + rho[ n ] * u1_n );
   const RealType rho_f_s = 0.5 * ( rho[ c ] * u1_c + rho[ s ] * u1_s );
   const RealType rho_g_e = 0.5 * ( rho[ c ] * u2_c + rho[ e ] * u2_e );
   const RealType rho_g_w = 0.5 * ( rho[ c ] * u2_c + rho[ w ] * u2_w );
   const RealType rho_g_n = 0.5 * ( rho[ c ] * u2_c + rho[ n ] * u2_n );
   const RealType rho_g_s = 0.5 * ( rho[ c ] * u2_c + rho[ s ] * u2_s );

   rho_t = - 1.0 / mu_D_c * ( rho_f_e * dy_e - rho_g_e * dx_e +
                              rho_f_n * dy_n - rho_g_n * dx_n +
                              rho_f_w * dy_w - rho_g_w * dx_w +
                              rho_f_s * dy_s - rho_g_s * dx_s )
           + 1.0 / ( 8.0 * mu_D_c ) *
                            ( ( mu_D_c + mu_D_e ) * ( rho[ e ] - rho[ c ] ) +
                              ( mu_D_c + mu_D_n ) * ( rho[ n ] - rho[ c ] ) +
                              ( mu_D_c + mu_D_w ) * ( rho[ w ] - rho[ c ] ) +
                              ( mu_D_c + mu_D_s ) * ( rho[ s ] - rho[ c ] ) );
}

template< typename Real,
          typename Device,
          typename Index,
          typename PressureGradient >
Real tnlLaxFridrichs< tnlGrid< 2, Real, Device, Index >, PressureGradient  > :: regularize( const Real& r ) const
{
   return r + ( ( r >= 0 ) - ( r < 0 ) ) * this -> regularizeEps;
}

#endif
