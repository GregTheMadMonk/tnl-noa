/***************************************************************************
                          laxFridrichs.h  -  description
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

#ifndef LAXFRIDRICHS_H_
#define LAXFRIDRICHS_H_

template< typename MeshType >
class laxFridrichs
{
   public:

   typedef typename MeshType :: RealType RealType;
   typedef typename MeshType :: DeviceType DeviceType;
   typedef typename MeshType :: IndexType IndexType;

   template< typename ConservativeVector,
             typename VelocityVector >
//             typename PressureVector >
   void getExplicitRhs( const MeshType& mesh,
                        const IndexType centralVolume,
                        const ConservativeVector& rho,
                        const ConservativeVector& rho_u1,
                        const ConservativeVector& rho_u2,
                        const VelocityVector& u1,
                        const VelocityVector& u2,
//                        const PressureVector& p,
                        ConservativeVector& rho_t,
                        ConservativeVector& rho_u1_t,
                        ConservativeVector& rho_u2_t,
                        const typename MeshType :: RealType viscosityCoeff = 1.0 ) const;
};

template< typename MeshType >
   template< typename ConservativeVector,
             typename VelocityVector >
//             typename PressureVector >
void laxFridrichs< MeshType > :: getExplicitRhs( const MeshType& mesh,
                                                 const IndexType centralVolume,
                                                 const ConservativeVector& rho,
                                                 const ConservativeVector& rho_u1,
                                                 const ConservativeVector& rho_u2,
                                                 const VelocityVector& u1,
                                                 const VelocityVector& u2,
                                                 ConservativeVector& rho_t,
                                                 ConservativeVector& rho_u1_t,
                                                 ConservativeVector& rho_u2_t,
                                                 const typename MeshType :: RealType viscosityCoeff ) const
{
   const IndexType& xSize = mesh. getDimensions(). x();
   const IndexType& ySize = mesh. getDimensions(). y();
   const RealType hx = mesh. getSpaceStep(). x();
   const RealType hy = mesh. getSpaceStep(). y();

   const IndexType& c = centralVolume;
   const IndexType e = mesh. getNodeNeighbour( centralVolume,  0,  1 );
   const IndexType w = mesh. getNodeNeighbour( centralVolume,  0, -1 );
   const IndexType n = mesh. getNodeNeighbour( centralVolume,  1,  0 );
   const IndexType s = mesh. getNodeNeighbour( centralVolume, -1,  0 );

   /****
    * rho_t + ( rho u_1 )_x + ( rho u_2 )_y =  0
    */
   rho_t[ c ]= viscosityCoeff * 0.25 * ( rho[ e ] + rho[ w ] + rho[ s ] + rho[ n ] - 4.0 * rho[ c ] )
                - ( rho[ e ] * u1[ e ] - rho[ w ] * u1[ w ] ) / ( 2.0 * hx )
                - ( rho[ n ] * u2[ n ] - rho[ s ] * u2[ s ] ) / ( 2.0 * hy );

    /****
     * ( rho * u1 )_t + ( rho * u1 * u1 )_x + ( rho * u1 * u2 )_y =  0
     */
    rho_u1_t[ c ] = viscosityCoeff * 0.25 * ( rho_u1[ e ] + rho_u1[ w ] + rho_u1[ s ] + rho_u1[ n ] - 4.0 * rho_u1[ c ] )
                    - ( rho_u1[ e ] * u1[ e ] - rho_u1[ w ] * u1[ w ] ) / ( 2.0 * hx )
                    - ( rho_u1[ n ] * u2[ n ] - rho_u1[ s ] * u2[ s ] ) / ( 2.0 * hy );
    rho_u2_t[ c ] = viscosityCoeff * 0.25 * ( rho_u2[ e ] + rho_u2[ w ] + rho_u2[ s ] + rho_u2[ n ] - 4.0 * rho_u2[ c ] )
                    - ( rho_u2[ e ] * u1[ e ] - rho_u2[ w ] * u1[ w ] ) / ( 2.0 * hx )
                    - ( rho_u2[ n ] * u2[ n ] - rho_u2[ s ] * u2[ s ] ) / ( 2.0 * hy );
}

#endif
