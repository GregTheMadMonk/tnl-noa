/***************************************************************************
                          tnlFDMGradientNorm.h  -  description
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

#ifndef TNLFDMGRADIENTNORM_H
#define	TNLFDMGRADIENTNORM_H

#include<operators/fdm/tnlForwardFiniteDifference.h>

template< typename Mesh,
          template< typename, int, int, int, typename, typename > class DifferenceOperatorTemplate = tnlForwardFiniteDifference,
          typename Real = typename Mesh::RealType,
          typename Index = typename Mesh::IndexType > 
class tnlFDMGradientNorm
{
};

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          template< typename, int, int, int, typename, typename > class DifferenceOperatorTemplate,
          typename Real,
          typename Index >
class tnlFDMGradientNorm< tnlGrid< 1,MeshReal, Device, MeshIndex >, DifferenceOperatorTemplate, Real, Index >
   : public tnlDomain< 1, MeshInteriorDomain >
{
   public: 
   
   typedef tnlGrid< 1, MeshReal, Device, MeshIndex > MeshType;
   typedef typename MeshType::CoordinatesType CoordinatesType;
   typedef Real RealType;
   typedef Device DeviceType;
   typedef Index IndexType;

   static tnlString getType()
   {
      return tnlString( "tnlFDMGradientNorm< " ) +
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
      DifferenceOperatorTemplate< typename MeshEntity::MeshType, 1, 0, 0, Real, Index > XDifferenceOperator;
      //const IndexType& cellIndex = entity.getIndex();
      //const typename MeshEntity::template NeighbourEntities< 1 >& neighbourEntities = entity.getNeighbourEntities();      
      //const typename MeshEntity::MeshType& mesh = entity.getMesh();
      const RealType u_x = XDifferenceOperator( u, entity );
      return sqrt( this->epsSquare + u_x * u_x );          
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
          template< typename, int, int, int, typename, typename > class DifferenceOperatorTemplate,
          typename Real,
          typename Index >
class tnlFDMGradientNorm< tnlGrid< 2,MeshReal, Device, MeshIndex >, DifferenceOperatorTemplate, Real, Index >
   : public tnlDomain< 2, MeshInteriorDomain >
{
   public: 
   
   typedef tnlGrid< 2, MeshReal, Device, MeshIndex > MeshType;
   typedef typename MeshType::CoordinatesType CoordinatesType;
   typedef Real RealType;
   typedef Device DeviceType;
   typedef Index IndexType;

   static tnlString getType()
   {
      return tnlString( "tnlFDMGradientNorm< " ) +
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
      DifferenceOperatorTemplate< typename MeshEntity::MeshType, 1, 0, 0, Real, Index > XDifferenceOperator;
      DifferenceOperatorTemplate< typename MeshEntity::MeshType, 0, 1, 0, Real, Index > YDifferenceOperator;
      //const IndexType& cellIndex = entity.getIndex();
      //const typename MeshEntity::template NeighbourEntities< 2 >& neighbourEntities = entity.getNeighbourEntities();      
      //const typename MeshEntity::MeshType& mesh = entity.getMesh();
      //const RealType& u_c = u[ cellIndex ];
      const RealType u_x = XDifferenceOperator( u, entity );
      const RealType u_y = YDifferenceOperator( u, entity );
      return sqrt( this->epsSquare + u_x * u_x + u_y * u_y );       
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
          template< typename, int, int, int, typename, typename > class DifferenceOperatorTemplate,
          typename Real,
          typename Index >
class tnlFDMGradientNorm< tnlGrid< 3, MeshReal, Device, MeshIndex >, DifferenceOperatorTemplate, Real, Index >
   : public tnlDomain< 3, MeshInteriorDomain >
{
   public: 
   
   typedef tnlGrid< 3, MeshReal, Device, MeshIndex > MeshType;
   typedef typename MeshType::CoordinatesType CoordinatesType;
   typedef Real RealType;
   typedef Device DeviceType;
   typedef Index IndexType;

   static tnlString getType()
   {
      return tnlString( "tnlFDMGradientNorm< " ) +
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
      DifferenceOperatorTemplate< typename MeshEntity::MeshType, 1, 0, 0, Real, Index > XDifferenceOperator;
      DifferenceOperatorTemplate< typename MeshEntity::MeshType, 0, 1, 0, Real, Index > YDifferenceOperator;
      DifferenceOperatorTemplate< typename MeshEntity::MeshType, 0, 0, 1, Real, Index > ZDifferenceOperator;
      //const IndexType& cellIndex = entity.getIndex();
      //const typename MeshEntity::template NeighbourEntities< 2 >& neighbourEntities = entity.getNeighbourEntities();      
      //const typename MeshEntity::MeshType& mesh = entity.getMesh();
      //const RealType& u_c = u[ cellIndex ];
      const RealType u_x = XDifferenceOperator( u, entity );
      const RealType u_y = YDifferenceOperator( u, entity );
      const RealType u_z = ZDifferenceOperator( u, entity );
      return sqrt( this->epsSquare + u_x * u_x + u_y * u_y + u_z * u_z );             
   }
   
        
   void setEps(const Real& eps)
   {
      this->epsSquare = eps*eps;
   }   
   
   private:
   
   RealType epsSquare;
};


#endif	/* TNLFDMGRADIENTNORM_H */

