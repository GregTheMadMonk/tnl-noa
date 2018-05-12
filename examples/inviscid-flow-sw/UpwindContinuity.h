/***************************************************************************
                          UpwindContinuity.h  -  description
                             -------------------
    begin                : Feb 17, 2017
    copyright            : (C) 2017 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */


#pragma once

#include <TNL/Containers/Vector.h>
#include <TNL/Meshes/Grid.h>
#include <TNL/Functions/VectorField.h>
#include <TNL/SharedPointer.h>

namespace TNL {

   
template< typename Mesh,
          typename Real = typename Mesh::RealType,
          typename Index = typename Mesh::IndexType >
class UpwindContinuityBase
{
   public:
      
      typedef Real RealType;
      typedef Index IndexType;
      typedef Mesh MeshType;
      typedef typename MeshType::DeviceType DeviceType;
      typedef typename MeshType::CoordinatesType CoordinatesType;
      typedef Functions::MeshFunction< MeshType > MeshFunctionType;
      static const int Dimensions = MeshType::getMeshDimension();
      typedef Functions::VectorField< Dimensions, MeshFunctionType > VelocityFieldType;
      typedef SharedPointer< MeshFunctionType > MeshFunctionPointer;
      typedef SharedPointer< VelocityFieldType > VelocityFieldPointer;

      static String getType()
      {
         return String( "UpwindContinuity< " ) +
             MeshType::getType() + ", " +
             TNL::getType< Real >() + ", " +
             TNL::getType< Index >() + " >"; 
      }

      void setTau(const Real& tau)
      {
          this->tau = tau;
      };

      void setGamma(const Real& gamma)
      {
          this->gamma = gamma;
      };
      
      void setPressure( const MeshFunctionPointer& pressure )
      {
          this->pressure = pressure;
      };
      
      void setVelocity( const VelocityFieldPointer& velocity )
      {
          this->velocity = velocity;
      };

      RealType positiveDensityFlux( const RealType& density, const RealType& velocity, const RealType& pressure ) const
      {
         const RealType& speedOfSound = std::sqrt( this->gamma * pressure / density );
         const RealType& machNumber = velocity / speedOfSound;
         if ( machNumber <= -1.0 )
            return 0.0;
        else if ( machNumber <= 0.0 )
            return density * speedOfSound / ( 2 * this->gamma ) * ( machNumber + 1.0 );
        else if ( machNumber <= 1.0 )
            return density * speedOfSound / ( 2 * this->gamma ) * ( ( 2.0 * this->gamma - 1.0 ) * machNumber + 1.0 );
        else 
            return density * velocity;
      };

      RealType negativeDensityFlux( const RealType& density, const RealType& velocity, const RealType& pressure ) const
      {
         const RealType& speedOfSound = std::sqrt( this->gamma * pressure / density );
         const RealType& machNumber = velocity / speedOfSound;
         if ( machNumber <= -1.0 )
            return density * velocity;
        else if ( machNumber <= 0.0 )
            return density * speedOfSound / ( 2 * this->gamma ) * ( ( 2.0 * this->gamma - 1.0 ) * machNumber - 1.0 );
        else if ( machNumber <= 1.0 )
            return density * speedOfSound / ( 2 * this->gamma ) * ( machNumber - 1.0 );
        else 
            return 0.0;
      };
      
      RealType multiply (const RealType& a, const RealType& b ) const
      {
         return a * b;
      };
      

      protected:
         
         RealType tau;

         RealType gamma;
         
         VelocityFieldPointer velocity;

         MeshFunctionPointer pressure;
         
};

   
template< typename Mesh,
          typename Real = typename Mesh::RealType,
          typename Index = typename Mesh::IndexType >
class UpwindContinuity
{
};



template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
class UpwindContinuity< Meshes::Grid< 1, MeshReal, Device, MeshIndex >, Real, Index >
   : public UpwindContinuityBase< Meshes::Grid< 1, MeshReal, Device, MeshIndex >, Real, Index >
{
   public:
      typedef Meshes::Grid< 1, MeshReal, Device, MeshIndex > MeshType;
      typedef UpwindContinuityBase< MeshType, Real, Index > BaseType;
      
      using typename BaseType::RealType;
      using typename BaseType::IndexType;
      using typename BaseType::DeviceType;
      using typename BaseType::CoordinatesType;
      using typename BaseType::MeshFunctionType;
      using typename BaseType::VelocityFieldType;
      using typename BaseType::VelocityFieldPointer;
      using BaseType::Dimensions;

      template< typename MeshFunction, typename MeshEntity >
      __cuda_callable__
      Real operator()( const MeshFunction& u,
                       const MeshEntity& entity,
                       const RealType& time = 0.0 ) const
      {
         static_assert( MeshEntity::getEntityDimension() == 1, "Wrong mesh entity dimensions." ); 
         static_assert( MeshFunction::getEntitiesDimension() == 1, "Wrong preimage function" ); 
         const typename MeshEntity::template NeighborEntities< 1 >& neighborEntities = entity.getNeighborEntities(); 

         const RealType& hxInverse = entity.getMesh().template getSpaceStepsProducts< -1 >();
 
         const IndexType& center = entity.getIndex(); 
         const IndexType& east   = neighborEntities.template getEntityIndex< 1 >(); 
         const IndexType& west   = neighborEntities.template getEntityIndex< -1 >();

         const RealType& pressure_center = this->pressure.template getData< DeviceType >()[ center ];
         const RealType& pressure_west   = this->pressure.template getData< DeviceType >()[ west ];
         const RealType& pressure_east   = this->pressure.template getData< DeviceType >()[ east ];

         const RealType& velocity_x_center = this->velocity.template getData< DeviceType >()[ 0 ].template getData< DeviceType >()[ center ];
         const RealType& velocity_x_west   = this->velocity.template getData< DeviceType >()[ 0 ].template getData< DeviceType >()[ west ];
         const RealType& velocity_x_east   = this->velocity.template getData< DeviceType >()[ 0 ].template getData< DeviceType >()[ east ];

         return -hxInverse * (
                                   this->positiveDensityFlux( u[ center ], velocity_x_center, pressure_center )
                                -  this->positiveDensityFlux( u[ west   ], velocity_x_west  , pressure_west   )
                                -  this->negativeDensityFlux( u[ center ], velocity_x_center, pressure_center )
                                +  this->negativeDensityFlux( u[ east   ], velocity_x_east  , pressure_east   )
//                                  u[ center ] / ( 2 * this->gamma ) * ( ( 2 * this->gamma - 1 ) * velocity_x_center + std::sqrt( this->multiply( this->gamma, pressure_center ) / u[ center ] ) )
//                                - u[ west   ] / ( 2 * this->gamma ) * ( ( 2 * this->gamma - 1 ) * velocity_x_west   + std::sqrt( this->gamma * pressure_west   / u[ west ]   ) )
//                                - u[ center ] / ( 2 * this->gamma ) * ( velocity_x_center - std::sqrt( this->gamma * pressure_center / u[ center ] ) ) 
//                                + u[ east   ] / ( 2 * this->gamma ) * ( velocity_x_east   - std::sqrt( this->gamma * pressure_east   / u[ east   ] ) )
                             );
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
class UpwindContinuity< Meshes::Grid< 2, MeshReal, Device, MeshIndex >, Real, Index >
   : public UpwindContinuityBase< Meshes::Grid< 2, MeshReal, Device, MeshIndex >, Real, Index >
{
   public:
      typedef Meshes::Grid< 2, MeshReal, Device, MeshIndex > MeshType;
      typedef UpwindContinuityBase< MeshType, Real, Index > BaseType;
      
      using typename BaseType::RealType;
      using typename BaseType::IndexType;
      using typename BaseType::DeviceType;
      using typename BaseType::CoordinatesType;
      using typename BaseType::MeshFunctionType;
      using typename BaseType::VelocityFieldType;
      using typename BaseType::VelocityFieldPointer;
      using BaseType::Dimensions;      

      template< typename MeshFunction, typename MeshEntity >
      __cuda_callable__
      Real operator()( const MeshFunction& u,
                       const MeshEntity& entity,
                       const RealType& time = 0.0 ) const
      {
         static_assert( MeshEntity::getEntityDimension() == 2, "Wrong mesh entity dimensions." ); 
         static_assert( MeshFunction::getEntitiesDimension() == 2, "Wrong preimage function" ); 
         const typename MeshEntity::template NeighborEntities< 2 >& neighborEntities = entity.getNeighborEntities(); 

         //rho
         const RealType& hxInverse = entity.getMesh().template getSpaceStepsProducts< -1, 0 >(); 
         const RealType& hyInverse = entity.getMesh().template getSpaceStepsProducts< 0, -1 >(); 

         const IndexType& center = entity.getIndex(); 
         const IndexType& east   = neighborEntities.template getEntityIndex<  1,  0 >(); 
         const IndexType& west   = neighborEntities.template getEntityIndex< -1,  0 >(); 
         const IndexType& north  = neighborEntities.template getEntityIndex<  0,  1 >(); 
         const IndexType& south  = neighborEntities.template getEntityIndex<  0, -1 >();

         const RealType& pressure_center = this->pressure.template getData< DeviceType >()[ center ];
         const RealType& pressure_west   = this->pressure.template getData< DeviceType >()[ west ];
         const RealType& pressure_east   = this->pressure.template getData< DeviceType >()[ east ];
         const RealType& pressure_north  = this->pressure.template getData< DeviceType >()[ north ];
         const RealType& pressure_south  = this->pressure.template getData< DeviceType >()[ south ];

         const RealType& velocity_x_center = this->velocity.template getData< DeviceType >()[ 0 ].template getData< DeviceType >()[ center ];
         const RealType& velocity_x_west   = this->velocity.template getData< DeviceType >()[ 0 ].template getData< DeviceType >()[ west ];
         const RealType& velocity_x_east   = this->velocity.template getData< DeviceType >()[ 0 ].template getData< DeviceType >()[ east ];

         const RealType& velocity_y_center = this->velocity.template getData< DeviceType >()[ 1 ].template getData< DeviceType >()[ center ];
         const RealType& velocity_y_north  = this->velocity.template getData< DeviceType >()[ 1 ].template getData< DeviceType >()[ north ];
         const RealType& velocity_y_south  = this->velocity.template getData< DeviceType >()[ 1 ].template getData< DeviceType >()[ south ];
         
         return -hxInverse * ( 
                                  u[ center ] / ( 2 * this->gamma ) * ( ( 2 * this->gamma - 1 ) * velocity_x_center + std::sqrt( this->gamma * pressure_center / u[ center ] ) )
                                - u[ west   ] / ( 2 * this->gamma ) * ( ( 2 * this->gamma - 1 ) * velocity_x_west   + std::sqrt( this->gamma * pressure_west   / u[ west ]   ) )
                                - u[ center ] / ( 2 * this->gamma ) * ( velocity_x_center - std::sqrt( this->gamma * pressure_center / u[ center ] ) )
                                + u[ east   ] / ( 2 * this->gamma ) * ( velocity_x_east   - std::sqrt( this->gamma * pressure_east   / u[ east   ] ) )
                             )
                -hyInverse * ( 
                                  u[ center ] / ( 2 * this->gamma ) * ( ( 2 * this->gamma - 1 ) * velocity_y_center + std::sqrt( this->gamma * pressure_center / u[ center ] ) )
                                - u[ south  ] / ( 2 * this->gamma ) * ( ( 2 * this->gamma - 1 ) * velocity_y_south  + std::sqrt( this->gamma * pressure_south  / u[ south ]  ) )
                                - u[ center ] / ( 2 * this->gamma ) * ( velocity_y_center - std::sqrt( this->gamma * pressure_center / u[ center ] ) )
                                + u[ north  ] / ( 2 * this->gamma ) * ( velocity_y_north  - std::sqrt( this->gamma * pressure_north  / u[ north ]  ) ) 
                             ); 
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
class UpwindContinuity< Meshes::Grid< 3, MeshReal, Device, MeshIndex >, Real, Index >
   : public UpwindContinuityBase< Meshes::Grid< 3, MeshReal, Device, MeshIndex >, Real, Index >
{
   public:
      typedef Meshes::Grid< 3, MeshReal, Device, MeshIndex > MeshType;
      typedef UpwindContinuityBase< MeshType, Real, Index > BaseType;
      
      using typename BaseType::RealType;
      using typename BaseType::IndexType;
      using typename BaseType::DeviceType;
      using typename BaseType::CoordinatesType;
      using typename BaseType::MeshFunctionType;
      using typename BaseType::VelocityFieldType;
      using typename BaseType::VelocityFieldPointer;
      using BaseType::Dimensions;

      template< typename MeshFunction, typename MeshEntity >
      __cuda_callable__
      Real operator()( const MeshFunction& u,
                       const MeshEntity& entity,
                       const RealType& time = 0.0 ) const
      {
         static_assert( MeshEntity::getEntityDimension() == 3, "Wrong mesh entity dimensions." ); 
         static_assert( MeshFunction::getEntitiesDimension() == 3, "Wrong preimage function" ); 
         const typename MeshEntity::template NeighborEntities< 3 >& neighborEntities = entity.getNeighborEntities(); 

         //rho
         const RealType& hxInverse = entity.getMesh().template getSpaceStepsProducts< -1,  0,  0 >(); 
         const RealType& hyInverse = entity.getMesh().template getSpaceStepsProducts<  0, -1,  0 >(); 
         const RealType& hzInverse = entity.getMesh().template getSpaceStepsProducts<  0,  0, -1 >();
 
         const IndexType& center = entity.getIndex(); 
         const IndexType& east   = neighborEntities.template getEntityIndex<  1,  0,  0 >(); 
         const IndexType& west   = neighborEntities.template getEntityIndex< -1,  0,  0 >(); 
         const IndexType& north  = neighborEntities.template getEntityIndex<  0,  1,  0 >(); 
         const IndexType& south  = neighborEntities.template getEntityIndex<  0, -1,  0 >();
         const IndexType& up     = neighborEntities.template getEntityIndex<  0,  0,  1 >(); 
         const IndexType& down   = neighborEntities.template getEntityIndex<  0,  0, -1 >();

         const RealType& pressure_center = this->pressure.template getData< DeviceType >()[ center ];
         const RealType& pressure_west   = this->pressure.template getData< DeviceType >()[ west ];
         const RealType& pressure_east   = this->pressure.template getData< DeviceType >()[ east ];
         const RealType& pressure_north  = this->pressure.template getData< DeviceType >()[ north ];
         const RealType& pressure_south  = this->pressure.template getData< DeviceType >()[ south ];
         const RealType& pressure_up     = this->pressure.template getData< DeviceType >()[ up ];
         const RealType& pressure_down   = this->pressure.template getData< DeviceType >()[ down ];
         
         const RealType& velocity_x_center = this->velocity.template getData< DeviceType >()[ 0 ].template getData< DeviceType >()[ center ];
         const RealType& velocity_x_west   = this->velocity.template getData< DeviceType >()[ 0 ].template getData< DeviceType >()[ west ];
         const RealType& velocity_x_east   = this->velocity.template getData< DeviceType >()[ 0 ].template getData< DeviceType >()[ east ];

         const RealType& velocity_y_center = this->velocity.template getData< DeviceType >()[ 1 ].template getData< DeviceType >()[ center ];
         const RealType& velocity_y_north  = this->velocity.template getData< DeviceType >()[ 1 ].template getData< DeviceType >()[ north ];
         const RealType& velocity_y_south  = this->velocity.template getData< DeviceType >()[ 1 ].template getData< DeviceType >()[ south ];

         const RealType& velocity_z_center = this->velocity.template getData< DeviceType >()[ 2 ].template getData< DeviceType >()[ center ];
         const RealType& velocity_z_up     = this->velocity.template getData< DeviceType >()[ 2 ].template getData< DeviceType >()[ up ];
         const RealType& velocity_z_down   = this->velocity.template getData< DeviceType >()[ 2 ].template getData< DeviceType >()[ down ];
         
         return -hxInverse * ( 
                                  u[ center ] / ( 2 * this->gamma ) * ( ( 2 * this->gamma - 1 ) * velocity_x_center + std::sqrt( this->gamma * pressure_center / u[ center ] ) )
                                - u[ west ]   / ( 2 * this->gamma ) * ( ( 2 * this->gamma - 1 ) * velocity_x_west   + std::sqrt( this->gamma * pressure_west   / u[ west ]   ) )
                                - u[ center ] / ( 2 * this->gamma ) * ( velocity_x_center - std::sqrt( this->gamma * pressure_center / u[ center ] ) )
                                + u[ east ]   / ( 2 * this->gamma ) * ( velocity_x_east   - std::sqrt( this->gamma * pressure_east   / u[ east   ] ) )
                             )
                -hyInverse * ( 
                                  u[ center ] / ( 2 * this->gamma ) * ( ( 2 * this->gamma - 1 ) * velocity_y_center + std::sqrt( this->gamma * pressure_center / u[ center ] ) )
                                - u[ south ]  / ( 2 * this->gamma ) * ( ( 2 * this->gamma - 1 ) * velocity_y_south  + std::sqrt( this->gamma * pressure_south  / u[ south ]  ) )
                                - u[ center ] / ( 2 * this->gamma ) * ( velocity_y_center - std::sqrt( this->gamma * pressure_center / u[ center ] ) )
                                + u[ north ]  / ( 2 * this->gamma ) * ( velocity_y_north  - std::sqrt( this->gamma * pressure_north  / u[ north ]  ) )
                             )
                -hzInverse * ( 
                                  u[ center ] / ( 2 * this->gamma ) * ( ( 2 * this->gamma - 1 ) * velocity_z_center + std::sqrt( this->gamma * pressure_center / u[ center ] ) )
                                - u[ down ]   / ( 2 * this->gamma ) * ( ( 2 * this->gamma - 1 ) * velocity_z_down   + std::sqrt( this->gamma * pressure_down   / u[ down ]   ) )
                                - u[ center ] / ( 2 * this->gamma ) * ( velocity_z_center - std::sqrt( this->gamma * pressure_center / u[ center ] ) ) 
                                + u[ up ]     / ( 2 * this->gamma ) * ( velocity_z_up     - std::sqrt( this->gamma * pressure_up     / u[ up ]     ) ) 
                             );
         
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


} //namespace TNL
