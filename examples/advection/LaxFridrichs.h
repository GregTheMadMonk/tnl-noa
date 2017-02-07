#ifndef LaxFridrichs_H
#define LaxFridrichs_H

#include <TNL/Containers/Vector.h>
#include <TNL/Meshes/Grid.h>
#include <TNL/Functions/VectorField.h>

namespace TNL {

template< typename Mesh,
          typename Real = typename Mesh::RealType,
          typename Index = typename Mesh::IndexType,
          typename VelocityFunction = Functions::MeshFunction< Mesh > >
class LaxFridrichs
{
};

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index,
          typename VelocityFunction >
class LaxFridrichs< Meshes::Grid< 1, MeshReal, Device, MeshIndex >, Real, Index, VelocityFunction >
{
   public:
      
      typedef Meshes::Grid< 1, MeshReal, Device, MeshIndex > MeshType;
      static const int Dimensions = MeshType::getMeshDimensions();
      typedef typename MeshType::CoordinatesType CoordinatesType;
      typedef Real RealType;
      typedef Device DeviceType;
      typedef Index IndexType;
      typedef Functions::MeshFunction< MeshType > MeshFunctionType;
      typedef VelocityFunction VelocityFunctionType;
      typedef Functions::VectorField< Dimensions, VelocityFunctionType > VelocityFieldType;
      
      LaxFridrichs() : artificialViscosity( 1.0 ) {}
      
      bool setup( const Config::ParameterContainer& parameters,
                  const String& prefix = "" )
      {
         return true;
      }

      static String getType();
      
      void setViscosity(const Real& artificalViscosity)
      {
         this->artificialViscosity = artificalViscosity;
      }
      
      void setTau(const Real& tau)
      {
          this->tau = tau;
      };

      VelocityFieldType& getVelocityField()
      {
         return this->velocityField;
      }
      
      const VelocityFieldType& getVelocityField() const
      {
         return this->velocityField;
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
         typedef Functions::FunctionAdapter< MeshType, VelocityFunctionType > FunctionAdapter;
         return ( 0.5 / this->tau ) * this->artificialViscosity * ( u[ west ]- 2.0 * u[ center ] + u[ east ] ) -
                FunctionAdapter::getValue( this->velocityField[ 0 ], entity, time ) * ( u[ east ] - u[west] ) * hxInverse * 0.5;
      }
      
   protected:
            
      RealType tau;
      
      RealType artificialViscosity;
      
      VelocityFieldType velocityField;
};

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index,
          typename VelocityFunction >
class LaxFridrichs< Meshes::Grid< 2, MeshReal, Device, MeshIndex >, Real, Index, VelocityFunction >
{
   public:
      
      typedef Meshes::Grid< 2, MeshReal, Device, MeshIndex > MeshType;
      static const int Dimensions = MeshType::getMeshDimensions();
      typedef typename MeshType::CoordinatesType CoordinatesType;
      typedef Real RealType;
      typedef Device DeviceType;
      typedef Index IndexType;
      typedef Functions::MeshFunction< MeshType > MeshFunctionType;
      typedef VelocityFunction VelocityFunctionType;
      typedef Functions::VectorField< Dimensions, VelocityFunctionType > VelocityFieldType;
      
      LaxFridrichs()
         : artificialViscosity( 1.0 ) {}      
      
      bool setup( const Config::ParameterContainer& parameters,
                  const String& prefix = "" )
      {
         return true;
      }

      static String getType();
      
      void setViscosity(const Real& artificalViscosity)
      {
         this->artificialViscosity = artificalViscosity;
      }
      
      void setTau(const Real& tau)
      {
          this->tau = tau;
      };

      VelocityFieldType& getVelocityField()
      {
         return this->velocityField;
      }
      
      const VelocityFieldType& getVelocityField() const
      {
         return this->velocityField;
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
         
         typedef Functions::FunctionAdapter< MeshType, VelocityFunctionType > FunctionAdapter;
         return ( 0.25 / this->tau ) * this->artificialViscosity * ( u[ west ] + u[ east ] + u[ north ] + u[ south ] - 4.0 * u[ center ] ) -
                0.5 * ( FunctionAdapter::getValue( this->velocityField[ 0 ], entity, time ) * ( u[ east ] - u[ west ] ) * hxInverse +
                        FunctionAdapter::getValue( this->velocityField[ 1 ], entity, time ) * ( u[ north ] - u[ south ] ) * hyInverse );         
      }
      
   protected:
            
      RealType tau;
      
      RealType artificialViscosity;
      
      VelocityFieldType velocityField;
};

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index,
          typename VelocityFunction >
class LaxFridrichs< Meshes::Grid< 3, MeshReal, Device, MeshIndex >, Real, Index, VelocityFunction >
{
   public:
      
      typedef Meshes::Grid< 3, MeshReal, Device, MeshIndex > MeshType;
      static const int Dimensions = MeshType::getMeshDimensions();
      typedef typename MeshType::CoordinatesType CoordinatesType;
      typedef Real RealType;
      typedef Device DeviceType;
      typedef Index IndexType;
      typedef Functions::MeshFunction< MeshType > MeshFunctionType;
      typedef VelocityFunction VelocityFunctionType;
      typedef Functions::VectorField< Dimensions, VelocityFunctionType > VelocityFieldType;
      
      LaxFridrichs()
         : artificialViscosity( 1.0 ) {}      
      
      bool setup( const Config::ParameterContainer& parameters,
                  const String& prefix = "" )
      {
         return true;
      }

      static String getType();
      
      void setViscosity(const Real& artificalViscosity)
      {
         this->artificialViscosity = artificalViscosity;
      }
      
      void setTau(const Real& tau)
      {
          this->tau = tau;
      };

      VelocityFieldType& getVelocityField()
      {
         return this->velocityField;
      }
      
      const VelocityFieldType& getVelocityField() const
      {
         return this->velocityField;
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

         const RealType& hxInverse = entity.getMesh().template getSpaceStepsProducts< -1,  0,  0 >(); 
         const RealType& hyInverse = entity.getMesh().template getSpaceStepsProducts<  0, -1,  0 >(); 
         const RealType& hzInverse = entity.getMesh().template getSpaceStepsProducts<  0,  0, -1 >(); 
         const IndexType& center = entity.getIndex();
         const IndexType& east  = neighbourEntities.template getEntityIndex<  1,  0,  0 >(); 
         const IndexType& west  = neighbourEntities.template getEntityIndex< -1,  0,  0 >(); 
         const IndexType& north = neighbourEntities.template getEntityIndex<  0,  1,  0 >(); 
         const IndexType& south = neighbourEntities.template getEntityIndex<  0, -1,  0 >(); 
         const IndexType& up    = neighbourEntities.template getEntityIndex<  0,  0,  1 >(); 
         const IndexType& down  = neighbourEntities.template getEntityIndex<  0,  0, -1 >(); 
         
         typedef Functions::FunctionAdapter< MeshType, VelocityFunctionType > FunctionAdapter;
         return ( 0.25 / this->tau ) * this->artificialViscosity * ( u[ west ] + u[ east ] + u[ north ] + u[ south ] + u[ up ] + u[ down ] - 6.0 * u[ center ] ) -
                0.5 * ( FunctionAdapter::getValue( this->velocityField[ 0 ], entity, time ) * ( u[ east ] - u[ west ] ) * hxInverse +
                        FunctionAdapter::getValue( this->velocityField[ 1 ], entity, time ) * ( u[ north ] - u[ south ] ) * hyInverse +
                        FunctionAdapter::getValue( this->velocityField[ 2 ], entity, time ) * ( u[ up ] - u[ down ] ) * hzInverse );         
      }
      
   protected:
            
      RealType tau;
      
      RealType artificialViscosity;
      
      VelocityFieldType velocityField;
};


} // namespace TNL


//#include "LaxFridrichs_impl.h"

#endif	/* LaxFridrichs_H */
