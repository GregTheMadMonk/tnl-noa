/***************************************************************************
                          LaxFridrichsMomentumX.h  -  description
                             -------------------
    begin                : Feb 18, 2017
    copyright            : (C) 2017 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */


#pragma once

#include <TNL/Containers/Vector.h>
#include <TNL/Meshes/Grid.h>
#include "LaxFridrichsMomentumBase.h"

namespace TNL {

template< typename Mesh,
          typename Real = typename Mesh::RealType,
          typename Index = typename Mesh::IndexType >
class LaxFridrichsMomentumX
{
};

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
class LaxFridrichsMomentumX< Meshes::Grid< 1, MeshReal, Device, MeshIndex >, Real, Index >
   : public LaxFridrichsMomentumBase< Meshes::Grid< 1, MeshReal, Device, MeshIndex >, Real, Index >
{
   public:

      typedef Meshes::Grid< 1, MeshReal, Device, MeshIndex > MeshType;
      typedef LaxFridrichsMomentumBase< MeshType, Real, Index > BaseType;
      
      using typename BaseType::RealType;
      using typename BaseType::IndexType;
      using typename BaseType::DeviceType;
      using typename BaseType::CoordinatesType;
      using typename BaseType::MeshFunctionType;
      using typename BaseType::MeshFunctionPointer;
      using typename BaseType::VelocityFieldType;
      using typename BaseType::VelocityFieldPointer;
      using BaseType::Dimensions;
      
      static String getType()
      {
         return String( "LaxFridrichsMomentumX< " ) +
             MeshType::getType() + ", " +
             TNL::getType< Real >() + ", " +
             TNL::getType< Index >() + " >"; 
      }
      

      template< typename MeshFunction, typename MeshEntity >
      __cuda_callable__
      Real operator()( const MeshFunction& u,
                       const MeshEntity& entity,
                       const RealType& time = 0.0 ) const
      {
         static_assert( MeshEntity::entityDimensions == 1, "Wrong mesh entity dimensions." ); 
         static_assert( MeshFunction::getEntitiesDimensions() == 1, "Wrong preimage function" ); 
         const typename MeshEntity::template NeighbourEntities< 1 >& neighbourEntities = entity.getNeighbourEntities(); 

         const RealType& hxInverse = entity.getMesh().template getSpaceStepsProducts< -1 >(); 
         const IndexType& center = entity.getIndex(); 
         const IndexType& east = neighbourEntities.template getEntityIndex< 1 >(); 
         const IndexType& west = neighbourEntities.template getEntityIndex< -1 >();
         const RealType& pressure_west = this->pressure.template getData< DeviceType >()[ west ];
         const RealType& pressure_east = this->pressure.template getData< DeviceType >()[ east ];
         const RealType& velocity_x_east = this->velocity.template getData< DeviceType >()[ 0 ].template getData< DeviceType >()[ east ];
         const RealType& velocity_x_west = this->velocity.template getData< DeviceType >()[ 0 ].template getData< DeviceType >()[ west ];
         
         return 0.5 * ( this->tau * ( u[ west ]  + u[ east ]  - 2.0 * u[ center ] ) 
                        - ( ( u[ west ] * velocity_x_west + pressure_west ) 
                          - ( u[ east ] * velocity_x_east + pressure_east ) ) * hxInverse );
      }

      /*template< typename MeshEntity >
      __cuda_callable__
      Index getLinearSystemRowLength( const MeshType& mesh,
                                      const IndexType& index,
                                      const MeshEntity& entity ) const;

      template< typename MeshEntity, typename Vector, typename MatrixRow >
      __cuda_callable__
      void updateLinearSystem( const RealType& time,
                               const RealType& tau,
                               const MeshType& mesh,
                               const IndexType& index,
                               const MeshEntity& entity,
                               const MeshFunctionType& u,
                               Vector& b,
                               MatrixRow& matrixRow ) const;*/
};

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
class LaxFridrichsMomentumX< Meshes::Grid< 2, MeshReal, Device, MeshIndex >, Real, Index >
   : public LaxFridrichsMomentumBase< Meshes::Grid< 2, MeshReal, Device, MeshIndex >, Real, Index >
{
   public:
      typedef Meshes::Grid< 2, MeshReal, Device, MeshIndex > MeshType;
      typedef LaxFridrichsMomentumBase< MeshType, Real, Index > BaseType;
      
      using typename BaseType::RealType;
      using typename BaseType::IndexType;
      using typename BaseType::DeviceType;
      using typename BaseType::CoordinatesType;
      using typename BaseType::MeshFunctionType;
      using typename BaseType::MeshFunctionPointer;
      using typename BaseType::VelocityFieldType;
      using typename BaseType::VelocityFieldPointer;
      using BaseType::Dimensions;
      
      static String getType()
      {
         return String( "LaxFridrichsMomentumX< " ) +
             MeshType::getType() + ", " +
             TNL::getType< Real >() + ", " +
             TNL::getType< Index >() + " >"; 
      }      

      template< typename MeshFunction, typename MeshEntity >
      __cuda_callable__
      Real operator()( const MeshFunction& u,
                       const MeshEntity& entity,
                       const RealType& time = 0.0 ) const
      {
         static_assert( MeshEntity::entityDimensions == 2, "Wrong mesh entity dimensions." ); 
         static_assert( MeshFunction::getEntitiesDimensions() == 2, "Wrong preimage function" ); 
         const typename MeshEntity::template NeighbourEntities< 2 >& neighbourEntities = entity.getNeighbourEntities(); 


         const RealType& hxInverse = entity.getMesh().template getSpaceStepsProducts< -1, 0 >(); 
         const RealType& hyInverse = entity.getMesh().template getSpaceStepsProducts< 0, -1 >(); 
         const IndexType& center = entity.getIndex(); 
         const IndexType& east  = neighbourEntities.template getEntityIndex<  1,  0 >(); 
         const IndexType& west  = neighbourEntities.template getEntityIndex< -1,  0 >(); 
         const IndexType& north = neighbourEntities.template getEntityIndex<  0,  1 >(); 
         const IndexType& south = neighbourEntities.template getEntityIndex<  0, -1 >();
         
         const RealType& pressure_west = this->pressure.template getData< DeviceType >()[ west ];
         const RealType& pressure_east = this->pressure.template getData< DeviceType >()[ east ];
         const RealType& velocity_x_east = this->velocity.template getData< DeviceType >()[ 0 ].template getData< DeviceType >()[ east ];
         const RealType& velocity_x_west = this->velocity.template getData< DeviceType >()[ 0 ].template getData< DeviceType >()[ west ];
         const RealType& velocity_y_north = this->velocity.template getData< DeviceType >()[ 1 ].template getData< DeviceType >()[ north ];
         const RealType& velocity_y_south = this->velocity.template getData< DeviceType >()[ 1 ].template getData< DeviceType >()[ south ];         
         
         return 0.5 * ( this->tau * ( u[ west ] + u[ east ] + u[ south ] + u[ north ] - 4.0 * u[ center ] ) 
                       - ( ( u[ west ] * velocity_x_west + pressure_west )
                         - ( u[ east ] * velocity_x_east + pressure_east ) )* hxInverse
                       - ( ( u[ north ] * velocity_y_north )
                         - ( u[ south ] * velocity_y_south ) )* hyInverse );
      }

      /*template< typename MeshEntity >
      __cuda_callable__
      Index getLinearSystemRowLength( const MeshType& mesh,
                                      const IndexType& index,
                                      const MeshEntity& entity ) const;

      template< typename MeshEntity, typename Vector, typename MatrixRow >
      __cuda_callable__
      void updateLinearSystem( const RealType& time,
                               const RealType& tau,
                               const MeshType& mesh,
                               const IndexType& index,
                               const MeshEntity& entity,
                               const MeshFunctionType& u,
                               Vector& b,
                               MatrixRow& matrixRow ) const;*/
};

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
class LaxFridrichsMomentumX< Meshes::Grid< 3,MeshReal, Device, MeshIndex >, Real, Index >
   : public LaxFridrichsMomentumBase< Meshes::Grid< 3, MeshReal, Device, MeshIndex >, Real, Index >
{
   public:
      typedef Meshes::Grid< 3, MeshReal, Device, MeshIndex > MeshType;
      typedef LaxFridrichsMomentumBase< MeshType, Real, Index > BaseType;
      
      using typename BaseType::RealType;
      using typename BaseType::IndexType;
      using typename BaseType::DeviceType;
      using typename BaseType::CoordinatesType;
      using typename BaseType::MeshFunctionType;
      using typename BaseType::MeshFunctionPointer;
      using typename BaseType::VelocityFieldType;
      using typename BaseType::VelocityFieldPointer;
      using BaseType::Dimensions;      
      
      static String getType()
      {
         return String( "LaxFridrichsMomentumX< " ) +
             MeshType::getType() + ", " +
             TNL::getType< Real >() + ", " +
             TNL::getType< Index >() + " >"; 
      }      

      template< typename MeshFunction, typename MeshEntity >
      __cuda_callable__
      Real operator()( const MeshFunction& u,
                       const MeshEntity& entity,
                       const RealType& time = 0.0 ) const
      {
         static_assert( MeshEntity::entityDimensions == 3, "Wrong mesh entity dimensions." ); 
         static_assert( MeshFunction::getEntitiesDimensions() == 3, "Wrong preimage function" ); 
         const typename MeshEntity::template NeighbourEntities< 3 >& neighbourEntities = entity.getNeighbourEntities(); 
 
         const RealType& hxInverse = entity.getMesh().template getSpaceStepsProducts< -1, 0,  0 >(); 
         const RealType& hyInverse = entity.getMesh().template getSpaceStepsProducts< 0, -1,  0 >(); 
         const RealType& hzInverse = entity.getMesh().template getSpaceStepsProducts< 0,  0, -1 >(); 
         const IndexType& center = entity.getIndex(); 
         const IndexType& east  = neighbourEntities.template getEntityIndex<  1,  0,  0 >(); 
         const IndexType& west  = neighbourEntities.template getEntityIndex< -1,  0,  0 >(); 
         const IndexType& north = neighbourEntities.template getEntityIndex<  0,  1,  0 >(); 
         const IndexType& south = neighbourEntities.template getEntityIndex<  0, -1,  0 >();
         const IndexType& up    = neighbourEntities.template getEntityIndex<  0,  0,  1 >(); 
         const IndexType& down  = neighbourEntities.template getEntityIndex<  0,  0, -1 >();
         
         const RealType& pressure_west  = this->pressure.template getData< DeviceType >()[ west ];
         const RealType& pressure_east  = this->pressure.template getData< DeviceType >()[ east ];
         //const RealType& pressure_north = this->pressure.template getData< DeviceType >()[ north ];
         //const RealType& pressure_south = this->pressure.template getData< DeviceType >()[ south ];
         //const RealType& pressure_up    = this->pressure.template getData< DeviceType >()[ up ];
         //const RealType& pressure_down  = this->pressure.template getData< DeviceType >()[ down ];
         
         const RealType& velocity_x_east  = this->velocity.template getData< DeviceType >()[ 0 ].template getData< DeviceType >()[ east ];
         const RealType& velocity_x_west  = this->velocity.template getData< DeviceType >()[ 0 ].template getData< DeviceType >()[ west ];
         const RealType& velocity_y_north = this->velocity.template getData< DeviceType >()[ 1 ].template getData< DeviceType >()[ north ];
         const RealType& velocity_y_south = this->velocity.template getData< DeviceType >()[ 1 ].template getData< DeviceType >()[ south ];
         const RealType& velocity_z_up    = this->velocity.template getData< DeviceType >()[ 2 ].template getData< DeviceType >()[ up ];
         const RealType& velocity_z_down  = this->velocity.template getData< DeviceType >()[ 2 ].template getData< DeviceType >()[ down ];
         return 0.5 * ( this->tau * ( u[ west ] + u[ east ] + u[ south ] + u[ north ] + u[ up ] + u[ down ] - 6.0 * u[ center ] ) 
                       - ( ( u[ west ] * velocity_x_west + pressure_west )
                         - ( u[ east ] * velocity_x_east + pressure_east ) )* hxInverse
                       - ( ( u[ north ] * velocity_y_north )
                         - ( u[ south ] * velocity_y_south ) )* hyInverse
                       - ( ( u[ up ] * velocity_z_up )
                         - ( u[ down ] * velocity_z_down ) )* hzInverse );
      }

      /*template< typename MeshEntity >
      __cuda_callable__
      Index getLinearSystemRowLength( const MeshType& mesh,
                                      const IndexType& index,
                                      const MeshEntity& entity ) const;

      template< typename MeshEntity, typename Vector, typename MatrixRow >
      __cuda_callable__
      void updateLinearSystem( const RealType& time,
                               const RealType& tau,
                               const MeshType& mesh,
                               const IndexType& index,
                               const MeshEntity& entity,
                               const MeshFunctionType& u,
                               Vector& b,
                               MatrixRow& matrixRow ) const;*/
};


} // namespace TNL

