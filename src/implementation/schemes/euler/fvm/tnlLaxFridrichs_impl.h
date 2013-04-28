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
          typename Index >
template< typename PressureGradient >
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
          typename Index >
template< typename PressureGradient >
void tnlLaxFridrichs< tnlGrid< 2, Real, Device, Index >, PressureGradient  > :: bindMesh( const MeshType& mesh )
{
   this -> mesh = &mesh;
}


template< typename Real,
          typename Device,
          typename Index >
template< typename PressureGradient >
void tnlLaxFridrichs< tnlGrid< 2, Real, Device, Index >, PressureGradient  > :: setRegularization( const RealType& epsilon )
{
   this -> regularizeEps = epsilon;
}

template< typename Real,
          typename Device,
          typename Index >
template< typename PressureGradient >
void tnlLaxFridrichs< tnlGrid< 2, Real, Device, Index >, PressureGradient  > :: setViscosityCoefficient( const RealType& v )
{
   this -> viscosityCoefficient = v;
}

template< typename Real,
          typename Device,
          typename Index >
template< typename PressureGradient >
   template< typename Vector >
void tnlLaxFridrichs< tnlGrid< 2, Real, Device, Index >, PressureGradient  > :: setRho( Vector& rho )
{
   this -> rho. bind( rho );
}

template< typename Real,
          typename Device,
          typename Index >
template< typename PressureGradient >
   template< typename Vector >
void tnlLaxFridrichs< tnlGrid< 2, Real, Device, Index >, PressureGradient  > :: setRhoU1( Vector& rho_u1 )
{
   this -> rho_u1. bind( rho_u1 );
}

template< typename Real,
          typename Device,
          typename Index >
template< typename PressureGradient >
   template< typename Vector >
void tnlLaxFridrichs< tnlGrid< 2, Real, Device, Index >, PressureGradient  > :: setRhoU2( Vector& rho_u2 )
{
   this -> rho_u2. bind( rho_u2 );
}

template< typename Real,
          typename Device,
          typename Index >
template< typename PressureGradient >
   template< typename Vector >
void tnlLaxFridrichs< tnlGrid< 2, Real, Device, Index >, PressureGradient  > :: setPressureGradient( Vector& grad_p )
{
   this -> pressureGradient = &grad_p;
}

template< typename Real,
          typename Device,
          typename Index >
template< typename PressureGradient >
void tnlLaxFridrichs< tnlGrid< 2, Real, Device, Index >, PressureGradient  > :: getExplicitRhs( const IndexType centralVolume,
                                                                      RealType& rho_t,
                                                                      RealType& rho_u1_t,
                                                                      RealType& rho_u2_t ) const
{
   tnlAssert( mesh, cerr << "No mesh has been binded with the Lax-Fridrichs scheme." );
   tnlAssert( pressureGradient, cerr << "No pressure gradient was set in the the Lax-Fridrichs scheme." )

   const IndexType& xSize = this -> mesh -> getDimensions(). x();
   const IndexType& ySize = this -> mesh -> getDimensions(). y();
   const RealType hx = this -> mesh -> getSpaceStep(). x();
   const RealType hy = this -> mesh -> getSpaceStep(). y();

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
   RealType p_x, p_y;
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
}

template< typename Real,
          typename Device,
          typename Index >
template< typename PressureGradient >
typename MeshType :: RealType tnlLaxFridrichs< tnlGrid< 2, Real, Device, Index >, PressureGradient  > :: regularize( const RealType& r ) const
{
   return r + ( ( r >= 0 ) - ( r < 0 ) ) * this -> regularizeEps;
}

#endif
