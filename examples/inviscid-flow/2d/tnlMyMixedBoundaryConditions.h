/*** coppied and changed
/***************************************************************************
                          tnlMyMixedBoundaryConditions.h  -  description
                             -------------------
    begin                : Nov 17, 2014
    copyright            : (C) 2014 by oberhuber
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

#ifndef TNLMyMixedBOUNDARYCONDITIONS_H_
#define TNLMyMixedBOUNDARYCONDITIONS_H_

#include <operators/tnlOperator.h>
#include <functions/tnlConstantFunction.h>
#include <functions/tnlFunctionAdapter.h>

template< typename Mesh,
          typename Function = tnlConstantFunction< Mesh::getMeshDimensions(), typename Mesh::RealType >,
          int MeshEntitiesDimensions = Mesh::getMeshDimensions(),
          typename Real = typename Mesh::RealType,
          typename Index = typename Mesh::IndexType >
class tnlMyMixedBoundaryConditions
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

      void setX0( const RealType& x0 )
      {
         this->x0 = x0;
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
      const MeshType& mesh = entity.getMesh();
      const auto& neighbourEntities = entity.getNeighbourEntities();
      typedef typename MeshType::Cell Cell;
      int count = mesh.template getEntitiesCount< Cell >();
      count = sqrt(count);
      if( entity.getCoordinates().x() == 0 )
      {
         if ( entity.getCoordinates().y() < count * x0 )
            return u[ neighbourEntities.template getEntityIndex< 0, 0 >() ];
         else if( entity.getCoordinates().y() < count -1 )
            return u[ neighbourEntities.template getEntityIndex< 1, 0 >() ];
         else if( entity.getCoordinates().y() == count )
            return u[ neighbourEntities.template getEntityIndex< 0, 0 >() ];
      }
      else if( entity.getCoordinates().y() == 0 )
      {
         if ( entity.getCoordinates().x() < count * x0 )
            return u[ neighbourEntities.template getEntityIndex< 0, 0 >() ];
         else if( entity.getCoordinates().x() < count -1 )
            return u[ neighbourEntities.template getEntityIndex< 0, 1 >() ];
         else if( entity.getCoordinates().y() == count )
            return u[ neighbourEntities.template getEntityIndex< 0, 0 >() ];
      }
      else if( entity.getCoordinates().x() == count ) 
            return u[ neighbourEntities.template getEntityIndex< -1, 0 >() ];
      else if( entity.getCoordinates().y() == count ) 
            return u[ neighbourEntities.template getEntityIndex< 0, -1 >() ];
      else return u[ neighbourEntities.template getEntityIndex< 0, 0 >() ];
         //tady se asi delaji okrajove podminky
         //static_assert( EntityType::getDimensions() == MeshEntitiesDimensions, "Wrong mesh entity dimensions." );
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
      
      RealType x0 = 0.3;
   
   //static_assert( Device::DeviceType == Function::Device::DeviceType );
};

template< typename Mesh,
          typename Function >
ostream& operator << ( ostream& str, const tnlMyMixedBoundaryConditions< Mesh, Function >& bc )
{
   str << "MyMixed boundary conditions: vector = " << bc.getVector();
   return str;
}

#endif /* TNLMyMixedBOUNDARYCONDITIONS_H_ */
