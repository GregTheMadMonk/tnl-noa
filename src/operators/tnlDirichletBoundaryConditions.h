/***************************************************************************
                          tnlDirichletBoundaryConditions.h  -  description
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

#ifndef TNLDIRICHLETBOUNDARYCONDITIONS_H_
#define TNLDIRICHLETBOUNDARYCONDITIONS_H_

template< typename Mesh,
          typename Function,
          typename Real = typename Mesh::RealType,
          typename Index = typename Mesh::IndexType >
class tnlDirichletBoundaryConditions
{
   public:

   typedef Mesh MeshType;
   typedef Function FunctionType;
   typedef Real RealType;
   typedef typename MeshType::DeviceType DeviceType;
   typedef Index IndexType;
  
   typedef tnlVector< RealType, DeviceType, IndexType> DofVectorType;
   typedef typename MeshType::VertexType VertexType;

   static void configSetup( tnlConfigDescription& config,
                            const tnlString& prefix = "" );

   bool setup( const tnlParameterContainer& parameters,
               const tnlString& prefix = "" );

   void setFunction( const Function& function );
   
   Function& getFunction();

   const Function& getFunction() const;
   
   template< typename EntityType >
   __cuda_callable__
   const RealType getValue( const EntityType& entity,                            
                            DofVectorType& u,
                            const RealType& time = 0 ) const;
   
   template< typename EntityType >
   __cuda_callable__
   IndexType getLinearSystemRowLength( const MeshType& mesh,
                                       const IndexType& index,
                                       const EntityType& entity ) const;

   template< typename MatrixRow,
             typename EntityType >
   __cuda_callable__
      void updateLinearSystem( const RealType& time,
                               const MeshType& mesh,
                               const IndexType& index,
                               const EntityType& entity,
                               DofVectorType& u,
                               DofVectorType& b,
                               MatrixRow& matrixRow ) const;

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

#include <operators/tnlDirichletBoundaryConditions_impl.h>

#endif /* TNLDIRICHLETBOUNDARYCONDITIONS_H_ */
