#pragma once 

#include <functions/tnlFunctionAdapter.h>
#include <operators/tnlOperator.h>

template< typename Mesh,
          typename Function,
          typename Real = typename Mesh::RealType,
          typename Index = typename Mesh::IndexType >
class tnlNeumannBoundaryConditions
{

};

/****
 * Base
 */
template< typename Function >
class tnlNeumannBoundaryConditionsBase
{
   public:
      
      typedef Function FunctionType;

      static void configSetup( tnlConfigDescription& config,
                               const tnlString& prefix = "" )
      {
         Function::configSetup( config, prefix );
      };

      bool setup( const tnlParameterContainer& parameters,
                  const tnlString& prefix = "" )
      {
          return this->function.setup( parameters, prefix );
      };

      void setFunction( const FunctionType& function )
      {
         this->function = function;
      };
      
      FunctionType& getFunction()
      {
         return this->function;
      };

      const FunctionType& getFunction() const
      {
         return this->function;
      };

   protected:

      FunctionType function;

};

/****
 * 1D grid
 */
template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Function,
          typename Real,
          typename Index >
class tnlNeumannBoundaryConditions< tnlGrid< 1, MeshReal, Device, MeshIndex >, Function, Real, Index >
   : public tnlNeumannBoundaryConditionsBase< Function >,
     public tnlOperator< tnlGrid< 1, MeshReal, Device, MeshIndex >,
                         MeshBoundaryDomain,
                         1, 1,
                         Real,
                         Index >
{
   public:

   typedef tnlGrid< 1, MeshReal, Device, MeshIndex > MeshType;
   typedef Real RealType;
   typedef Device DeviceType;
   typedef Index IndexType;

   typedef Function FunctionType;
   typedef tnlVector< RealType, DeviceType, IndexType> DofVectorType;
   typedef tnlStaticVector< 1, RealType > VertexType;
   typedef typename MeshType::CoordinatesType CoordinatesType;
   typedef tnlNeumannBoundaryConditions< MeshType, Function, Real, Index > ThisType;
   typedef tnlNeumannBoundaryConditionsBase< Function > BaseType;

   template< typename EntityType,
             typename MeshFunction >
   __cuda_callable__
   const RealType operator()( const MeshFunction& u,
                              const EntityType& entity,   
                              const RealType& time = 0 ) const
   {
      const MeshType& mesh = entity.getMesh();
      const auto& neighbourEntities = entity.getNeighbourEntities();
      const IndexType& index = entity.getIndex();
      if( entity.getCoordinates().x() == 0 )
         return u[ neighbourEntities.template getEntityIndex< 1 >() ] + entity.getMesh().getSpaceSteps().x() * 
            tnlFunctionAdapter< MeshType, FunctionType >::getValue( this->function, entity, time );
      else
         return u[ neighbourEntities.template getEntityIndex< -1 >() ] + entity.getMesh().getSpaceSteps().x() * 
            tnlFunctionAdapter< MeshType, FunctionType >::getValue( this->function, entity, time );   

   }


   template< typename EntityType >
   __cuda_callable__
   Index getLinearSystemRowLength( const MeshType& mesh,
                                   const IndexType& index,
                                   const EntityType& entity ) const
   {
      return 2;
   }

      template< typename PreimageFunction,
                typename MeshEntity,
                typename Matrix,
                typename Vector >
      __cuda_callable__
      void setMatrixElements( const PreimageFunction& u,
                                     const MeshEntity& entity,
                                     const RealType& time,
                                     const RealType& tau,
                                     Matrix& matrix,
                                     Vector& b ) const
      {
         const auto& neighbourEntities = entity.getNeighbourEntities();
         const IndexType& index = entity.getIndex();
         typename Matrix::MatrixRow matrixRow = matrix.getRow( index );
         if( entity.getCoordinates().x() == 0 )
         {
            matrixRow.setElement( 0, index, 1.0 );
            matrixRow.setElement( 1, neighbourEntities.template getEntityIndex< 1 >(), -1.0 );
            b[ index ] = entity.getMesh().getSpaceSteps().x() * 
               tnlFunctionAdapter< MeshType, FunctionType >::getValue( this->function, entity, time );
         }
         else
         {
            matrixRow.setElement( 0, neighbourEntities.template getEntityIndex< -1 >(), -1.0 );
            matrixRow.setElement( 1, index, 1.0 );
            b[ index ] = entity.getMesh().getSpaceSteps().x() *
               tnlFunctionAdapter< MeshType, FunctionType >::getValue( this->function, entity, time );
         }         
      }
};

/****
 * 2D grid
 */
template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Function,
          typename Real,
          typename Index >
class tnlNeumannBoundaryConditions< tnlGrid< 2, MeshReal, Device, MeshIndex >, Function, Real, Index >
   : public tnlNeumannBoundaryConditionsBase< Function >,
     public tnlOperator< tnlGrid< 2, MeshReal, Device, MeshIndex >,
                         MeshBoundaryDomain,
                         2, 2,
                         Real,
                         Index >

{
   public:

      typedef tnlGrid< 2, MeshReal, Device, MeshIndex > MeshType;
      typedef Real RealType;
      typedef Device DeviceType;
      typedef Index IndexType;

      typedef Function FunctionType;
      typedef tnlVector< RealType, DeviceType, IndexType> DofVectorType;
      typedef tnlStaticVector< 2, RealType > VertexType;
      typedef typename MeshType::CoordinatesType CoordinatesType;
      typedef tnlNeumannBoundaryConditions< MeshType, Function, Real, Index > ThisType;
      typedef tnlNeumannBoundaryConditionsBase< Function > BaseType;


      template< typename EntityType,
                typename MeshFunction >
      __cuda_callable__
      const RealType operator()( const MeshFunction& u,
                                 const EntityType& entity,                            
                                 const RealType& time = 0 ) const
      {
         const MeshType& mesh = entity.getMesh();
         const auto& neighbourEntities = entity.getNeighbourEntities();
         const IndexType& index = entity.getIndex();
         if( entity.getCoordinates().x() == 0 )
         {
            return u[ neighbourEntities.template getEntityIndex< 1, 0 >() ] + entity.getMesh().getSpaceSteps().x() *
               tnlFunctionAdapter< MeshType, FunctionType >::getValue( this->function, entity, time );
         }
         if( entity.getCoordinates().x() == entity.getMesh().getDimensions().x() - 1 )
         {
            return u[ neighbourEntities.template getEntityIndex< -1, 0 >() ] + entity.getMesh().getSpaceSteps().x() *
               tnlFunctionAdapter< MeshType, FunctionType >::getValue( this->function, entity, time );
         }
         if( entity.getCoordinates().y() == 0 )
         {
            return u[ neighbourEntities.template getEntityIndex< 0, 1 >() ] + entity.getMesh().getSpaceSteps().y() *
               tnlFunctionAdapter< MeshType, FunctionType >::getValue( this->function, entity, time );
         }
         if( entity.getCoordinates().y() == entity.getMesh().getDimensions().y() - 1 )
         {
            return u[ neighbourEntities.template getEntityIndex< 0, -1 >() ] + entity.getMesh().getSpaceSteps().y() *
               tnlFunctionAdapter< MeshType, FunctionType >::getValue( this->function, entity, time );
         }         
      }

      template< typename EntityType >
      __cuda_callable__
      Index getLinearSystemRowLength( const MeshType& mesh,
                                      const IndexType& index,
                                      const EntityType& entity ) const
      {
         return 2;
      }

      template< typename PreimageFunction,
                typename MeshEntity,
                typename Matrix,
                typename Vector >
      __cuda_callable__
      void setMatrixElements( const PreimageFunction& u,
                              const MeshEntity& entity,
                              const RealType& time,
                              const RealType& tau,
                              Matrix& matrix,
                              Vector& b ) const
      {
         const auto& neighbourEntities = entity.getNeighbourEntities();
         const IndexType& index = entity.getIndex();
         typename Matrix::MatrixRow matrixRow = matrix.getRow( index );
         if( entity.getCoordinates().x() == 0 )
         {
            matrixRow.setElement( 0, index,                                                1.0 );
            matrixRow.setElement( 1, neighbourEntities.template getEntityIndex< 1, 0 >(), -1.0 );
            b[ index ] = entity.getMesh().getSpaceSteps().x() *
               tnlFunctionAdapter< MeshType, FunctionType >::getValue( this->function, entity, time );
         }
         if( entity.getCoordinates().x() == entity.getMesh().getDimensions().x() - 1 )
         {
            matrixRow.setElement( 0, neighbourEntities.template getEntityIndex< -1, 0 >(), -1.0 );
            matrixRow.setElement( 1, index,                                                 1.0 );
            b[ index ] = entity.getMesh().getSpaceSteps().x() *
               tnlFunctionAdapter< MeshType, FunctionType >::getValue( this->function, entity, time );
         }
         if( entity.getCoordinates().y() == 0 )
         {
            matrixRow.setElement( 0, index,                                                1.0 );
            matrixRow.setElement( 1, neighbourEntities.template getEntityIndex< 0, 1 >(), -1.0 );
            b[ index ] = entity.getMesh().getSpaceSteps().y() *
               tnlFunctionAdapter< MeshType, FunctionType >::getValue( this->function, entity, time );
         }
         if( entity.getCoordinates().y() == entity.getMesh().getDimensions().y() - 1 )
         {
            matrixRow.setElement( 0, neighbourEntities.template getEntityIndex< 0, -1 >(), -1.0 );
            matrixRow.setElement( 1, index,                                                 1.0 );
            b[ index ] = entity.getMesh().getSpaceSteps().y() *
               tnlFunctionAdapter< MeshType, FunctionType >::getValue( this->function, entity, time );
         }         
      }
};

/****
 * 3D grid
 */
template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Function,
          typename Real,
          typename Index >
class tnlNeumannBoundaryConditions< tnlGrid< 3, MeshReal, Device, MeshIndex >, Function, Real, Index >
   : public tnlNeumannBoundaryConditionsBase< Function >,
     public tnlOperator< tnlGrid< 3, MeshReal, Device, MeshIndex >,
                         MeshBoundaryDomain,
                         3, 3,
                         Real,
                         Index >
{
   public:

      typedef tnlGrid< 3, MeshReal, Device, MeshIndex > MeshType;
      typedef Real RealType;
      typedef Device DeviceType;
      typedef Index IndexType;

      typedef Function FunctionType;
      typedef tnlVector< RealType, DeviceType, IndexType> DofVectorType;
      typedef tnlStaticVector< 3, RealType > VertexType;
      typedef typename MeshType::CoordinatesType CoordinatesType;
      typedef tnlNeumannBoundaryConditions< MeshType, Function, Real, Index > ThisType;
      typedef tnlNeumannBoundaryConditionsBase< Function > BaseType;   

      template< typename EntityType,
                typename MeshFunction >
      __cuda_callable__
      const RealType operator()( const MeshFunction& u,
                                 const EntityType& entity,
                                 const RealType& time = 0 ) const
      {
         const MeshType& mesh = entity.getMesh();
         const auto& neighbourEntities = entity.getNeighbourEntities();
         const IndexType& index = entity.getIndex();
         if( entity.getCoordinates().x() == 0 )
         {
            return u[ neighbourEntities.template getEntityIndex< 1, 0, 0 >() ] + entity.getMesh().getSpaceSteps().x() *
               tnlFunctionAdapter< MeshType, FunctionType >::getValue( this->function, entity, time );
         }
         if( entity.getCoordinates().x() == entity.getMesh().getDimensions().x() - 1 )
         {
            return u[ neighbourEntities.template getEntityIndex< -1, 0, 0 >() ] + entity.getMesh().getSpaceSteps().x() *
               tnlFunctionAdapter< MeshType, FunctionType >::getValue( this->function, entity, time );
         }
         if( entity.getCoordinates().y() == 0 )
         {
            return u[ neighbourEntities.template getEntityIndex< 0, 1, 0 >() ] + entity.getMesh().getSpaceSteps().y() *
               tnlFunctionAdapter< MeshType, FunctionType >::getValue( this->function, entity, time );
         }
         if( entity.getCoordinates().y() == entity.getMesh().getDimensions().y() - 1 )
         {
            return u[ neighbourEntities.template getEntityIndex< 0, -1, 0 >() ] + entity.getMesh().getSpaceSteps().y() *
               tnlFunctionAdapter< MeshType, FunctionType >::getValue( this->function, entity, time );
         }
         if( entity.getCoordinates().z() == 0 )
         {
            return u[ neighbourEntities.template getEntityIndex< 0, 0, 1 >() ] + entity.getMesh().getSpaceSteps().z() *
               tnlFunctionAdapter< MeshType, FunctionType >::getValue( this->function, entity, time );
         }
         if( entity.getCoordinates().z() == entity.getMesh().getDimensions().z() - 1 )
         {
            return u[ neighbourEntities.template getEntityIndex< 0, 0, -1 >() ] + entity.getMesh().getSpaceSteps().z() *
               tnlFunctionAdapter< MeshType, FunctionType >::getValue( this->function, entity, time );
         }   
      }


      template< typename EntityType >
      __cuda_callable__
      Index getLinearSystemRowLength( const MeshType& mesh,
                                      const IndexType& index,
                                      const EntityType& entity ) const
      {
         return 2;
      }

      template< typename PreimageFunction,
                typename MeshEntity,
                typename Matrix,
                typename Vector >
      __cuda_callable__
      void setMatrixElements( const PreimageFunction& u,
                                     const MeshEntity& entity,
                                     const RealType& time,
                                     const RealType& tau,
                                     Matrix& matrix,
                                     Vector& b ) const
      {
         const auto& neighbourEntities = entity.getNeighbourEntities();
         const IndexType& index = entity.getIndex();
         typename Matrix::MatrixRow matrixRow = matrix.getRow( index );
         if( entity.getCoordinates().x() == 0 )
         {
            matrixRow.setElement( 0, index,                                                   1.0 );
            matrixRow.setElement( 1, neighbourEntities.template getEntityIndex< 1, 0, 0 >(), -1.0 );
            b[ index ] = entity.getMesh().getSpaceSteps().x() *
               tnlFunctionAdapter< MeshType, FunctionType >::getValue( this->function, entity, time );
         }
         if( entity.getCoordinates().x() == entity.getMesh().getDimensions().x() - 1 )
         {
            matrixRow.setElement( 0, neighbourEntities.template getEntityIndex< -1, 0, 0 >(), -1.0 );
            matrixRow.setElement( 1, index,                                                    1.0 );
            b[ index ] = entity.getMesh().getSpaceSteps().x() *
               tnlFunctionAdapter< MeshType, FunctionType >::getValue( this->function, entity, time );
         }
         if( entity.getCoordinates().y() == 0 )
         {
            matrixRow.setElement( 0, index,                                                   1.0 );
            matrixRow.setElement( 1, neighbourEntities.template getEntityIndex< 0, 1, 0 >(), -1.0 );
            b[ index ] = entity.getMesh().getSpaceSteps().y() * 
               tnlFunctionAdapter< MeshType, FunctionType >::getValue( this->function, entity, time );
         }
         if( entity.getCoordinates().y() == entity.getMesh().getDimensions().y() - 1 )
         {
            matrixRow.setElement( 0, neighbourEntities.template getEntityIndex< 0, -1, 0 >(), -1.0 );
            matrixRow.setElement( 1, index,                                                    1.0 );
            b[ index ] = entity.getMesh().getSpaceSteps().y() *
               tnlFunctionAdapter< MeshType, FunctionType >::getValue( this->function, entity, time );
         }
         if( entity.getCoordinates().z() == 0 )
         {
            matrixRow.setElement( 0, index,                                                   1.0 );
            matrixRow.setElement( 1, neighbourEntities.template getEntityIndex< 0, 0, 1 >(), -1.0 );
            b[ index ] = entity.getMesh().getSpaceSteps().z() *
               tnlFunctionAdapter< MeshType, FunctionType >::getValue( this->function, entity, time );
         }
         if( entity.getCoordinates().z() == entity.getMesh().getDimensions().z() - 1 )
         {
            matrixRow.setElement( 0, neighbourEntities.template getEntityIndex< 0, 0, -1 >(), -1.0 );
            matrixRow.setElement( 1, index,                                                    1.0 );
            b[ index ] = entity.getMesh().getSpaceSteps().z() *
               tnlFunctionAdapter< MeshType, FunctionType >::getValue( this->function, entity, time );
         }
      }
};

template< typename Mesh,
          typename Function,
          typename Real,
          typename Index >
ostream& operator << ( ostream& str, const tnlNeumannBoundaryConditions< Mesh, Function, Real, Index >& bc )
{
   str << "Neumann boundary conditions: function = " << bc.getFunction();
   return str;
}


//#include <operators/tnlNeumannBoundaryConditions_impl.h>


