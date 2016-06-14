/***************************************************************************
                          tnlCoFVMGradientNorm.h  -  description
                             -------------------
    begin                : Jan 21, 2016
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

#ifndef TNLCOFVMGRADIENTNORM_H
#define	TNLCOFVMGRADIENTNORM_H

#include <mesh/tnlGrid.h>
#include <operators/geometric/tnlExactGradientNorm.h>
#include <operators/interpolants/tnlMeshEntitiesInterpolants.h>
#include <operators/tnlOperator.h>
#include <operators/tnlOperatorComposition.h>

template< typename Mesh,
          int MeshEntityDimensions = Mesh::getMeshDimensions(),
          typename Real = typename Mesh::RealType,
          typename Index = typename Mesh::IndexType >
class tnlCoFVMGradientNorm
{   
};

template< int MeshDimensions,
          typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
class tnlCoFVMGradientNorm< tnlGrid< MeshDimensions, MeshReal, Device, MeshIndex >, MeshDimensions, Real, Index >
: public tnlOperatorComposition< 
   tnlMeshEntitiesInterpolants< tnlGrid< MeshDimensions, MeshReal, Device, MeshIndex >,
                                MeshDimensions - 1,
                                MeshDimensions >,
   tnlCoFVMGradientNorm< tnlGrid< MeshDimensions, MeshReal, Device, MeshIndex >, MeshDimensions - 1, Real, Index > >
{  
   public:
      typedef tnlGrid< MeshDimensions, MeshReal, Device, MeshIndex > MeshType;
      typedef typename MeshType::CoordinatesType CoordinatesType;
      typedef Real RealType;
      typedef Device DeviceType;
      typedef Index IndexType;
      typedef tnlCoFVMGradientNorm< MeshType, MeshDimensions - 1, Real, Index > InnerOperator;
      typedef tnlMeshEntitiesInterpolants< MeshType, MeshDimensions - 1, MeshDimensions > OuterOperator;
      typedef tnlOperatorComposition< OuterOperator, InnerOperator > BaseType;
      typedef tnlExactGradientNorm< MeshDimensions, RealType > ExactOperatorType;
      typedef tnlSharedPointer< MeshType > MeshPointer;
         
      tnlCoFVMGradientNorm( const OuterOperator& outerOperator,
                            InnerOperator& innerOperator,
                            const MeshPointer& mesh )
      : BaseType( outerOperator, innerOperator, mesh )
      {}
      
      static tnlString getType()
      {
         return tnlString( "tnlCoFVMGradientNorm< " ) +
            MeshType::getType() + ", " +
            tnlString( MeshDimensions ) + ", " +
            ::getType< Real >() + ", " +
            ::getType< Index >() + " >";
      }
      
      void setEps( const RealType& eps )
      {
         this->getInnerOperator().setEps( eps );
      }
      
      static constexpr int getPreimageEntitiesDimensions() { return MeshDimensions; };
      static constexpr int getImageEntitiesDimensions() { return MeshDimensions; };

};

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
class tnlCoFVMGradientNorm< tnlGrid< 1,MeshReal, Device, MeshIndex >, 0, Real, Index >
   : public tnlOperator< tnlGrid< 1,MeshReal, Device, MeshIndex >, MeshInteriorDomain, 1, 0, Real, Index >
{
   public: 
   
   typedef tnlGrid< 1, MeshReal, Device, MeshIndex > MeshType;
   typedef typename MeshType::CoordinatesType CoordinatesType;
   typedef Real RealType;
   typedef Device DeviceType;
   typedef Index IndexType;
   typedef tnlExactGradientNorm< 1, RealType > ExactOperatorType;
   
   constexpr static int getPreimageEntitiesDimensions() { return MeshType::getMeshDimensions(); };
   constexpr static int getImageEntitiesDimensions() { return MeshType::getMeshDimensions() - 1; };
   
   tnlCoFVMGradientNorm()
   : epsSquare( 0.0 ){}

   static tnlString getType()
   {
      return tnlString( "tnlCoFVMGradientNorm< " ) +
         MeshType::getType() + ", 0, " +
         ::getType< Real >() + ", " +
         ::getType< Index >() + " >";
   }

   template< typename MeshFunction, typename MeshEntity >
   __cuda_callable__
   Real operator()( const MeshFunction& u,
                    const MeshEntity& entity,
                    const Real& time = 0.0 ) const
   {
      static_assert( MeshFunction::getDimensions() == 1, 
         "The mesh function u must be stored on mesh cells.." );
      static_assert( MeshEntity::getDimensions() == 0,
         "The complementary finite volume gradient norm may be evaluated only on faces." );
      const typename MeshEntity::template NeighbourEntities< 1 >& neighbourEntities = entity.template getNeighbourEntities< 1 >();
      
      const RealType& hxDiv = entity.getMesh().template getSpaceStepsProducts< -1 >();      
      const RealType& u_x = ( u[ neighbourEntities.template getEntityIndex<  1 >() ] -
                              u[ neighbourEntities.template getEntityIndex< -1 >() ] ) * hxDiv;
      return sqrt( this->epsSquare + ( u_x * u_x ) );          
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
class tnlCoFVMGradientNorm< tnlGrid< 2, MeshReal, Device, MeshIndex >, 1, Real, Index >
   : public tnlOperator< tnlGrid< 2,MeshReal, Device, MeshIndex >, MeshInteriorDomain, 2, 1, Real, Index >
{
   public: 
   
   typedef tnlGrid< 2, MeshReal, Device, MeshIndex > MeshType;
   typedef typename MeshType::CoordinatesType CoordinatesType;
   typedef Real RealType;
   typedef Device DeviceType;
   typedef Index IndexType;
   typedef tnlExactGradientNorm< 2, RealType > ExactOperatorType;
   
   constexpr static int getPreimageEntitiesDimensions() { return MeshType::getMeshDimensions(); };
   constexpr static int getImageEntitiesDimensions() { return MeshType::getMeshDimensions() - 1; };
   
   tnlCoFVMGradientNorm()
   : epsSquare( 0.0 ){}


   static tnlString getType()
   {
      return tnlString( "tnlCoFVMGradientNorm< " ) +
         MeshType::getType() + ", 1, " +
         ::getType< Real >() + ", " +
         ::getType< Index >() + " >";

   }
      
   template< typename MeshFunction, typename MeshEntity >
   __cuda_callable__
   Real operator()( const MeshFunction& u,
                    const MeshEntity& entity,
                    const Real& time = 0.0 ) const
   {      
      static_assert( MeshFunction::getDimensions() == 2, 
         "The mesh function u must be stored on mesh cells.." );
      static_assert( MeshEntity::getDimensions() == 1,
         "The complementary finite volume gradient norm may be evaluated only on faces." );
      const typename MeshEntity::template NeighbourEntities< 2 >& neighbourEntities = entity.template getNeighbourEntities< 2 >();
      const RealType& hxDiv = entity.getMesh().template getSpaceStepsProducts< -1,  0 >();
      const RealType& hyDiv = entity.getMesh().template getSpaceStepsProducts<  0, -1 >();
      if( entity.getOrientation().x() != 0.0 )
      {
         const RealType u_x =
            ( u[ neighbourEntities.template getEntityIndex<  1, 0 >()] -
              u[ neighbourEntities.template getEntityIndex< -1, 0 >()] ) * hxDiv;         
         RealType u_y;
         if( entity.getCoordinates().y() > 0 )
         {
            if( entity.getCoordinates().y() < entity.getMesh().getDimensions().y() - 1 )
               u_y = 0.25 * 
                  ( u[ neighbourEntities.template getEntityIndex<  1,  1 >() ] + 
                    u[ neighbourEntities.template getEntityIndex< -1,  1 >() ] - 
                    u[ neighbourEntities.template getEntityIndex<  1, -1 >() ] -
                    u[ neighbourEntities.template getEntityIndex< -1, -1 >() ] ) * hyDiv;
            else // if( entity.getCoordinates().y() < entity.getMesh().getDimensions().y() - 1 )
               u_y = 0.5 * 
                  ( u[ neighbourEntities.template getEntityIndex<  1,  0 >() ] + 
                    u[ neighbourEntities.template getEntityIndex< -1,  0 >() ] - 
                    u[ neighbourEntities.template getEntityIndex<  1, -1 >() ] -
                    u[ neighbourEntities.template getEntityIndex< -1, -1 >() ] ) * hyDiv;
         }
         else // if( entity.getCoordinates().y() > 0 )
         {
            u_y = 0.5 * 
               ( u[ neighbourEntities.template getEntityIndex<  1,  1 >() ] + 
                 u[ neighbourEntities.template getEntityIndex< -1,  1 >() ] - 
                 u[ neighbourEntities.template getEntityIndex<  1,  0 >() ] -
                 u[ neighbourEntities.template getEntityIndex< -1,  0 >() ] ) * hyDiv;
         }
         return sqrt( this->epsSquare + u_x * u_x + u_y * u_y );
      }
      RealType u_x;
      if( entity.getCoordinates().x() > 0 )
      {
         if( entity.getCoordinates().x() < entity.getMesh().getDimensions().x() - 1 )
            u_x = 0.25 * 
            ( u[ neighbourEntities.template getEntityIndex<  1,  1 >() ] + 
              u[ neighbourEntities.template getEntityIndex<  1, -1 >() ] - 
              u[ neighbourEntities.template getEntityIndex< -1,  1 >() ] -
              u[ neighbourEntities.template getEntityIndex< -1, -1 >() ] ) * hxDiv;
         else // if( entity.getCoordinates().x() < entity.getMesh().getDimensions().x() - 1 )
            u_x = 0.5 * 
            ( u[ neighbourEntities.template getEntityIndex<  0,  1 >() ] + 
              u[ neighbourEntities.template getEntityIndex<  0, -1 >() ] - 
              u[ neighbourEntities.template getEntityIndex< -1,  1 >() ] -
              u[ neighbourEntities.template getEntityIndex< -1, -1 >() ] ) * hxDiv;
      }
      else // if( entity.getCoordinates().x() > 0 )
      {
         u_x = 0.5 * 
            ( u[ neighbourEntities.template getEntityIndex<  1,  1 >() ] + 
              u[ neighbourEntities.template getEntityIndex<  1, -1 >() ] - 
              u[ neighbourEntities.template getEntityIndex<  0,  1 >() ] -
              u[ neighbourEntities.template getEntityIndex<  0, -1 >() ] ) * hxDiv;
      }
      const RealType u_y =
         ( u[ neighbourEntities.template getEntityIndex< 0,  1 >()] -
           u[ neighbourEntities.template getEntityIndex< 0, -1 >()] ) * hyDiv;
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
          typename Real,
          typename Index >
class tnlCoFVMGradientNorm< tnlGrid< 3, MeshReal, Device, MeshIndex >, 2, Real, Index >
   : public tnlOperator< tnlGrid< 3, MeshReal, Device, MeshIndex >, MeshInteriorDomain, 3, 2, Real, Index >
{
   public: 
   
   typedef tnlGrid< 3, MeshReal, Device, MeshIndex > MeshType;
   typedef typename MeshType::CoordinatesType CoordinatesType;
   typedef Real RealType;
   typedef Device DeviceType;
   typedef Index IndexType;
   typedef tnlExactGradientNorm< 3, RealType > ExactOperatorType;
   
   constexpr static int getPreimageEntitiesDimensions() { return MeshType::getMeshDimensions(); };
   constexpr static int getImageEntitiesDimensions() { return MeshType::getMeshDimensions() - 1; };
   
   tnlCoFVMGradientNorm()
   : epsSquare( 0.0 ){}   

   static tnlString getType()
   {
      return tnlString( "tnlCoFVMGradientNorm< " ) +
         MeshType::getType() + ", 2, " +
         ::getType< Real >() + ", " +
         ::getType< Index >() + " >";      
   }

   template< typename MeshFunction, typename MeshEntity >
   __cuda_callable__
   Real operator()( const MeshFunction& u,
                    const MeshEntity& entity,
                    const Real& time = 0.0 ) const
   {
      static_assert( MeshFunction::getDimensions() == 3, 
         "The mesh function u must be stored on mesh cells.." );
      static_assert( MeshEntity::getDimensions() == 2,
         "The complementary finite volume gradient norm may be evaluated only on faces." );
      const typename MeshEntity::template NeighbourEntities< 3 >& neighbourEntities = entity.template getNeighbourEntities< 3 >();
      const RealType& hxDiv = entity.getMesh().template getSpaceStepsProducts< -1,  0,  0 >();
      const RealType& hyDiv = entity.getMesh().template getSpaceStepsProducts<  0, -1,  0 >();
      const RealType& hzDiv = entity.getMesh().template getSpaceStepsProducts<  0,  0, -1 >();
      if( entity.getOrientation().x() != 0.0 )
      {
         const RealType u_x =
            ( u[ neighbourEntities.template getEntityIndex<  1,  0,  0 >()] -
              u[ neighbourEntities.template getEntityIndex< -1,  0,  0 >()] ) * hxDiv;         
         RealType u_y;
         if( entity.getCoordinates().y() > 0 )
         {
            if( entity.getCoordinates().y() < entity.getMesh().getDimensions().y() - 1 )
            {
               u_y = 0.25 * 
               ( u[ neighbourEntities.template getEntityIndex<  1,  1,  0 >() ] + 
                 u[ neighbourEntities.template getEntityIndex< -1,  1,  0 >() ] - 
                 u[ neighbourEntities.template getEntityIndex<  1, -1,  0 >() ] -
                 u[ neighbourEntities.template getEntityIndex< -1, -1,  0 >() ] ) * hyDiv;
            }
            else // if( entity.getCoordinates().y() < entity.getMesh().getDimensions().y() - 1 )
            {
               u_y = 0.5 * 
               ( u[ neighbourEntities.template getEntityIndex<  1,  0,  0 >() ] + 
                 u[ neighbourEntities.template getEntityIndex< -1,  0,  0 >() ] - 
                 u[ neighbourEntities.template getEntityIndex<  1, -1,  0 >() ] -
                 u[ neighbourEntities.template getEntityIndex< -1, -1,  0 >() ] ) * hyDiv;

            }
         }
         else // if( entity.getCoordinates().y() > 0 )
         {
            u_y = 0.5 * 
            ( u[ neighbourEntities.template getEntityIndex<  1,  1,  0 >() ] + 
              u[ neighbourEntities.template getEntityIndex< -1,  1,  0 >() ] - 
              u[ neighbourEntities.template getEntityIndex<  1,  0,  0 >() ] -
              u[ neighbourEntities.template getEntityIndex< -1,  0,  0 >() ] ) * hyDiv;

         }
         RealType u_z;
         if( entity.getCoordinates().z() > 0 )
         {
            if( entity.getCoordinates().z() < entity.getMesh().getDimensions().z() - 1 )
            {
               u_z = 0.25 * 
               ( u[ neighbourEntities.template getEntityIndex<  1,  0,  1 >() ] + 
                 u[ neighbourEntities.template getEntityIndex< -1,  0,  1 >() ] - 
                 u[ neighbourEntities.template getEntityIndex<  1,  0, -1 >() ] -
                 u[ neighbourEntities.template getEntityIndex< -1,  0, -1 >() ] ) * hzDiv;
            }
            else //if( entity.getCoordinates().z() < entity.getMesh().getDimensions().z() - 1 )
            {
               u_z = 0.5 * 
               ( u[ neighbourEntities.template getEntityIndex<  1,  0,  0 >() ] + 
                 u[ neighbourEntities.template getEntityIndex< -1,  0,  0 >() ] - 
                 u[ neighbourEntities.template getEntityIndex<  1,  0, -1 >() ] -
                 u[ neighbourEntities.template getEntityIndex< -1,  0, -1 >() ] ) * hzDiv;
            }
         }
         else //if( entity.getCoordinates().z() > 0 )
         {
            u_z = 0.5 * 
            ( u[ neighbourEntities.template getEntityIndex<  1,  0,  1 >() ] + 
              u[ neighbourEntities.template getEntityIndex< -1,  0,  1 >() ] - 
              u[ neighbourEntities.template getEntityIndex<  1,  0,  0 >() ] -
              u[ neighbourEntities.template getEntityIndex< -1,  0,  0 >() ] ) * hzDiv;
         }
         return sqrt( this->epsSquare + u_x * u_x + u_y * u_y + u_z * u_z );
      }
      if( entity.getOrientation().y() != 0.0 )
      {
         RealType u_x;
         if( entity.getCoordinates().x() > 0 )
         {
            if( entity.getCoordinates().x() < entity.getMesh().getDimensions().x() - 1 )
            {
               u_x = 0.25 * 
               ( u[ neighbourEntities.template getEntityIndex<  1,  1,  0 >() ] + 
                 u[ neighbourEntities.template getEntityIndex<  1, -1,  0 >() ] - 
                 u[ neighbourEntities.template getEntityIndex< -1,  1,  0 >() ] -
                 u[ neighbourEntities.template getEntityIndex< -1, -1,  0 >() ] ) * hxDiv;
            }
            else // if( entity.getCoordinates().x() < entity.getMesh().getDimensions().x() - 1 )
            {
               u_x = 0.5 * 
               ( u[ neighbourEntities.template getEntityIndex<  0,  1,  0 >() ] + 
                 u[ neighbourEntities.template getEntityIndex<  0, -1,  0 >() ] - 
                 u[ neighbourEntities.template getEntityIndex< -1,  1,  0 >() ] -
                 u[ neighbourEntities.template getEntityIndex< -1, -1,  0 >() ] ) * hxDiv;
            }
         }
         else // if( entity.getCoordinates().x() > 0 )
         {
            u_x = 0.5 * 
            ( u[ neighbourEntities.template getEntityIndex<  1,  1,  0 >() ] + 
              u[ neighbourEntities.template getEntityIndex<  1, -1,  0 >() ] - 
              u[ neighbourEntities.template getEntityIndex<  0,  1,  0 >() ] -
              u[ neighbourEntities.template getEntityIndex<  0, -1,  0 >() ] ) * hxDiv;
         }
         const RealType u_y =
            ( u[ neighbourEntities.template getEntityIndex<  0,  1,  0 >()] -
              u[ neighbourEntities.template getEntityIndex<  0, -1,  0 >()] ) * hyDiv;
         RealType u_z;
         if( entity.getCoordinates().z() > 0 )
         {
            if( entity.getCoordinates().z() < entity.getMesh().getDimensions().z() - 1 )
            {
               u_z = 0.25 * 
               ( u[ neighbourEntities.template getEntityIndex<  0,  1,  1 >() ] + 
                 u[ neighbourEntities.template getEntityIndex<  0, -1,  1 >() ] - 
                 u[ neighbourEntities.template getEntityIndex<  0,  1, -1 >() ] -
                 u[ neighbourEntities.template getEntityIndex<  0, -1, -1 >() ] ) * hzDiv;
            }
            else // if( entity.getCoordinates().z() < entity.getMesh().getDimensions().z() - 1 )
            {
               u_z = 0.5 * 
               ( u[ neighbourEntities.template getEntityIndex<  0,  1,  0 >() ] + 
                 u[ neighbourEntities.template getEntityIndex<  0, -1,  0 >() ] - 
                 u[ neighbourEntities.template getEntityIndex<  0,  1, -1 >() ] -
                 u[ neighbourEntities.template getEntityIndex<  0, -1, -1 >() ] ) * hzDiv;
            }
         }
         else // if( entity.getCoordinates().z() > 0 )
         {
            u_z = 0.5 * 
            ( u[ neighbourEntities.template getEntityIndex<  0,  1,  1 >() ] + 
              u[ neighbourEntities.template getEntityIndex<  0, -1,  1 >() ] - 
              u[ neighbourEntities.template getEntityIndex<  0,  1,  0 >() ] -
              u[ neighbourEntities.template getEntityIndex<  0, -1,  0 >() ] ) * hzDiv;
         }
         return sqrt( this->epsSquare + u_x * u_x + u_y * u_y + u_z * u_z );
      }
      RealType u_x;
      if( entity.getCoordinates().x() > 0 )
      {
         if( entity.getCoordinates().x() < entity.getMesh().getDimensions().x() - 1 )
         {
            u_x = 0.25 * 
            ( u[ neighbourEntities.template getEntityIndex<  1,  0,  1 >() ] + 
              u[ neighbourEntities.template getEntityIndex<  1,  0, -1 >() ] - 
              u[ neighbourEntities.template getEntityIndex< -1,  0,  1 >() ] -
              u[ neighbourEntities.template getEntityIndex< -1,  0, -1 >() ] ) * hxDiv;
         }
         else // if( entity.getCoordinates().x() < entity.getMesh().getDimensions().x() - 1 )
         {
            u_x = 0.5 * 
            ( u[ neighbourEntities.template getEntityIndex<  0,  0,  1 >() ] + 
              u[ neighbourEntities.template getEntityIndex<  0,  0, -1 >() ] - 
              u[ neighbourEntities.template getEntityIndex< -1,  0,  1 >() ] -
              u[ neighbourEntities.template getEntityIndex< -1,  0, -1 >() ] ) * hxDiv;

         }
      }
      else // if( entity.getCoordinates().x() > 0 )
      {
         u_x = 0.5 * 
         ( u[ neighbourEntities.template getEntityIndex<  1,  0,  1 >() ] + 
           u[ neighbourEntities.template getEntityIndex<  1,  0, -1 >() ] - 
           u[ neighbourEntities.template getEntityIndex<  0,  0,  1 >() ] -
           u[ neighbourEntities.template getEntityIndex<  0,  0, -1 >() ] ) * hxDiv;         
      }
      RealType u_y;
      if( entity.getCoordinates().y() > 0 )
      {
         if( entity.getCoordinates().y() < entity.getMesh().getDimensions().y() - 1 )
         {      
            u_y = 0.25 * 
            ( u[ neighbourEntities.template getEntityIndex<  0,  1,  1 >() ] + 
              u[ neighbourEntities.template getEntityIndex<  0,  1, -1 >() ] - 
              u[ neighbourEntities.template getEntityIndex<  0, -1,  1 >() ] -
              u[ neighbourEntities.template getEntityIndex<  0, -1, -1 >() ] ) * hyDiv;
         }
         else //if( entity.getCoordinates().y() < entity.getMesh().getDimensions().y() - 1 )
         {
            u_y = 0.5 * 
            ( u[ neighbourEntities.template getEntityIndex<  0,  0,  1 >() ] + 
              u[ neighbourEntities.template getEntityIndex<  0,  0, -1 >() ] - 
              u[ neighbourEntities.template getEntityIndex<  0, -1,  1 >() ] -
              u[ neighbourEntities.template getEntityIndex<  0, -1, -1 >() ] ) * hyDiv;
         }
      }
      else //if( entity.getCoordinates().y() > 0 )
      {
         u_y = 0.5 * 
         ( u[ neighbourEntities.template getEntityIndex<  0,  1,  1 >() ] + 
           u[ neighbourEntities.template getEntityIndex<  0,  1, -1 >() ] - 
           u[ neighbourEntities.template getEntityIndex<  0,  0,  1 >() ] -
           u[ neighbourEntities.template getEntityIndex<  0,  0, -1 >() ] ) * hyDiv;         
      }
      const RealType u_z =
         ( u[ neighbourEntities.template getEntityIndex<  0,  0,  1 >()] -
           u[ neighbourEntities.template getEntityIndex<  0,  0, -1 >()] ) * hzDiv;
      return sqrt( this->epsSquare + u_x * u_x + u_y * u_y + u_z * u_z );
   }
   
        
   void setEps(const Real& eps)
   {
      this->epsSquare = eps*eps;
   }   
   
   private:
   
   RealType epsSquare;
};

#endif	/* TNLCOFVMGRADIENTNORM_H */

