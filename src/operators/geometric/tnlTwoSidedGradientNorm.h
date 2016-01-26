/***************************************************************************
                          tnlTwoSidedGradientNorm.h  -  description
                             -------------------
    begin                : Jan 11, 2016
    copyright            : (C) 2016 by Tomas Oberhuber
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

#ifndef TNLTWOSIDEDGRADIENTNORM_H
#define	TNLTWOSIDEDGRADIENTNORM_H

#include<operators/fdm/tnlForwardFiniteDifference.h>
#include<operators/fdm/tnlBackwardFiniteDifference.h>

template< typename Mesh,
          typename Real = typename Mesh::RealType,
          typename Index = typename Mesh::IndexType > 
class tnlTwoSidedGradientNorm
{
};

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
class tnlTwoSidedGradientNorm< tnlGrid< 1,MeshReal, Device, MeshIndex >, Real, Index >
   : public tnlDomain< 1, MeshInteriorDomain >
{
   public: 
   
   typedef tnlGrid< 1, MeshReal, Device, MeshIndex > MeshType;
   typedef typename MeshType::CoordinatesType CoordinatesType;
   typedef Real RealType;
   typedef Device DeviceType;
   typedef Index IndexType;
   
   tnlTwoSidedGradientNorm()
   : epsSquare( 0.0 ){}

   static tnlString getType()
   {
      return tnlString( "tnlTwoSidedGradientNorm< " ) +
         MeshType::getType() + ", " +
         ::getType< Real >() + ", " +
         ::getType< Index >() + " >";
   }

   template< typename MeshFunction, typename MeshEntity >
   __cuda_callable__
   Real operator()( const MeshFunction& u,
                    const MeshEntity& entity,
                    const Real& time = 0.0 ) const
   {
      tnlForwardFiniteDifference< typename MeshEntity::MeshType, 1, 0, 0, Real, Index > XForwardDifference;
      tnlBackwardFiniteDifference< typename MeshEntity::MeshType, 1, 0, 0, Real, Index > XBackwardDifference;
      const RealType u_x_f = XForwardDifference( u, entity );
      const RealType u_x_b = XBackwardDifference( u, entity );
      return sqrt( this->epsSquare + 0.5 * ( u_x_f * u_x_f + u_x_b * u_x_b ) );          
   }
                
   void setEps( const Real& eps )
   {
      this->epsSquare = eps*eps;
   }
      
   private:
   
   RealType epsSquare;
};


template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
class tnlTwoSidedGradientNorm< tnlGrid< 2,MeshReal, Device, MeshIndex >, Real, Index >
   : public tnlDomain< 2, MeshInteriorDomain >
{
   public: 
   
   typedef tnlGrid< 2, MeshReal, Device, MeshIndex > MeshType;
   typedef typename MeshType::CoordinatesType CoordinatesType;
   typedef Real RealType;
   typedef Device DeviceType;
   typedef Index IndexType;
   
   tnlTwoSidedGradientNorm()
   : epsSquare( 0.0 ){}


   static tnlString getType()
   {
      return tnlString( "tnlTwoSidedGradientNorm< " ) +
         MeshType::getType() + ", " +
         ::getType< Real >() + ", " +
         ::getType< Index >() + " >";

   }
      
   template< typename MeshFunction, typename MeshEntity >
   __cuda_callable__
   Real operator()( const MeshFunction& u,
                    const MeshEntity& entity,
                    const Real& time = 0.0 ) const
   {      
      tnlForwardFiniteDifference< typename MeshEntity::MeshType, 1, 0, 0, Real, Index > XForwardDifference;
      tnlForwardFiniteDifference< typename MeshEntity::MeshType, 0, 1, 0, Real, Index > YForwardDifference;
      tnlBackwardFiniteDifference< typename MeshEntity::MeshType, 1, 0, 0, Real, Index > XBackwardDifference;
      tnlBackwardFiniteDifference< typename MeshEntity::MeshType, 0, 1, 0, Real, Index > YBackwardDifference;
      const RealType u_x_f = XForwardDifference( u, entity );
      const RealType u_x_b = XBackwardDifference( u, entity );
      const RealType u_y_f = YForwardDifference( u, entity );
      const RealType u_y_b = YBackwardDifference( u, entity );
      
      return sqrt( this->epsSquare +
         0.5 * ( u_x_f * u_x_f + u_x_b * u_x_b + 
                 u_y_f * u_y_f + u_y_b * u_y_b ) );          
   }
           
   void setEps( const Real& eps )
   {
      this->epsSquare = eps*eps;
   }   
   
   
   private:
   
   RealType epsSquare;
};


template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
class tnlTwoSidedGradientNorm< tnlGrid< 3, MeshReal, Device, MeshIndex >, Real, Index >
   : public tnlDomain< 3, MeshInteriorDomain >
{
   public: 
   
   typedef tnlGrid< 3, MeshReal, Device, MeshIndex > MeshType;
   typedef typename MeshType::CoordinatesType CoordinatesType;
   typedef Real RealType;
   typedef Device DeviceType;
   typedef Index IndexType;
   
   tnlTwoSidedGradientNorm()
   : epsSquare( 0.0 ){}   

   static tnlString getType()
   {
      return tnlString( "tnlTwoSidedGradientNorm< " ) +
         MeshType::getType() + ", " +
         ::getType< Real >() + ", " +
         ::getType< Index >() + " >";      
   }

   template< typename MeshFunction, typename MeshEntity >
   __cuda_callable__
   Real operator()( const MeshFunction& u,
                    const MeshEntity& entity,
                    const Real& time = 0.0 ) const
   {
      tnlForwardFiniteDifference< typename MeshEntity::MeshType, 1, 0, 0, Real, Index > XForwardDifference;
      tnlForwardFiniteDifference< typename MeshEntity::MeshType, 0, 1, 0, Real, Index > YForwardDifference;
      tnlForwardFiniteDifference< typename MeshEntity::MeshType, 0, 0, 1, Real, Index > ZForwardDifference;
      tnlBackwardFiniteDifference< typename MeshEntity::MeshType, 1, 0, 0, Real, Index > XBackwardDifference;
      tnlBackwardFiniteDifference< typename MeshEntity::MeshType, 0, 1, 0, Real, Index > YBackwardDifference;
      tnlBackwardFiniteDifference< typename MeshEntity::MeshType, 0, 0, 1, Real, Index > ZBackwardDifference;
      const RealType u_x_f = XForwardDifference( u, entity );
      const RealType u_x_b = XBackwardDifference( u, entity );
      const RealType u_y_f = YForwardDifference( u, entity );
      const RealType u_y_b = YBackwardDifference( u, entity );
      const RealType u_z_f = ZForwardDifference( u, entity );
      const RealType u_z_b = ZBackwardDifference( u, entity );
      
      return sqrt( this->epsSquare +
         0.5 * ( u_x_f * u_x_f + u_x_b * u_x_b +
                 u_y_f * u_y_f + u_y_b * u_y_b +
                 u_z_f * u_z_f + u_z_b * u_z_b ) );          
      
   }
   
        
   void setEps(const Real& eps)
   {
      this->epsSquare = eps*eps;
   }   
   
   private:
   
   RealType epsSquare;
};


#endif	/* TNLTWOSIDEDGRADIENTNORM_H */

