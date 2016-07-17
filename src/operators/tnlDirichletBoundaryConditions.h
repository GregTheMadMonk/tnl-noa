/***************************************************************************
                          tnlDirichletBoundaryConditions.h  -  description
                             -------------------
    begin                : Nov 17, 2014
    copyright            : (C) 2014 by oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <operators/tnlOperator.h>
#include <functions/tnlConstantFunction.h>
#include <functions/tnlFunctionAdapter.h>

namespace TNL {

template< typename Mesh,
          typename Function = tnlConstantFunction< Mesh::getMeshDimensions(), typename Mesh::RealType >,
          int MeshEntitiesDimensions = Mesh::getMeshDimensions(),
          typename Real = typename Mesh::RealType,
          typename Index = typename Mesh::IndexType >
class tnlDirichletBoundaryConditions
: public tnlOperator< Mesh,
                      MeshBoundaryDomain,
                      MeshEntitiesDimensions,
                      MeshEntitiesDimensions,
                      Real,
                      Index >
{
   public:

      typedef Mesh MeshType;
      typedef Function FunctionType;
      typedef Real RealType;
      typedef typename MeshType::DeviceType DeviceType;
      typedef Index IndexType;

      typedef tnlVector< RealType, DeviceType, IndexType> DofVectorType;
      typedef typename MeshType::VertexType VertexType;

      static constexpr int getMeshDimensions() { return MeshType::meshDimensions; }

      static void configSetup( tnlConfigDescription& config,
                               const tnlString& prefix = "" )
      {
         Function::configSetup( config, prefix );
      }
 
      bool setup( const tnlParameterContainer& parameters,
                  const tnlString& prefix = "" )
      {
         return this->function.setup( parameters, prefix );
      }

      void setFunction( const Function& function )
      {
         this->function = function;
      }

      Function& getFunction()
      {
         return this->function;
      }
 
      const Function& getFunction() const
      {
         return this->function;
      }

      template< typename EntityType,
                typename MeshFunction >
      __cuda_callable__
      const RealType operator()( const MeshFunction& u,
                                 const EntityType& entity,
                                 const RealType& time = 0 ) const
      {
         //static_assert( EntityType::getDimensions() == MeshEntitiesDimensions, "Wrong mesh entity dimensions." );
         return tnlFunctionAdapter< MeshType, Function >::template getValue( this->function, entity, time );
      }

      template< typename EntityType >
      __cuda_callable__
      IndexType getLinearSystemRowLength( const MeshType& mesh,
                                          const IndexType& index,
                                          const EntityType& entity ) const
      {
         return 1;
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
         typename Matrix::MatrixRow matrixRow = matrix.getRow( entity.getIndex() );
         const IndexType& index = entity.getIndex();
         matrixRow.setElement( 0, index, 1.0 );
         b[ index ] = tnlFunctionAdapter< MeshType, Function >::getValue( this->function, entity, time );
      }
 

   protected:

      Function function;
 
   //static_assert( Device::DeviceType == Function::Device::DeviceType );
};

template< typename Mesh,
          typename Function >
ostream& operator << ( ostream& str, const tnlDirichletBoundaryConditions< Mesh, Function >& bc )
{
   str << "Dirichlet boundary conditions: vector = " << bc.getVector();
   return str;
}

} // namespace TNL
