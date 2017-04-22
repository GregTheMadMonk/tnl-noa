/***************************************************************************
                          tnlDirichletBoundaryConditions_impl.h  -  description
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

#ifndef TNLDIRICHLETBOUNDARYCONDITIONS_IMPL_H_
#define TNLDIRICHLETBOUNDARYCONDITIONS_IMPL_H_

#include <functions/tnlFunctionAdapter.h>

template< typename Mesh,
          typename Function,
          int MeshEntitiesDimension,
          typename Real,
          typename Index >
void
tnlDirichletBoundaryConditions< Mesh, Function, MeshEntitiesDimension, Real, Index >::
configSetup( tnlConfigDescription& config,
             const String& prefix )
{
   Function::configSetup( config );
}

template< typename Mesh,
          typename Function,
          int MeshEntitiesDimension,
          typename Real,
          typename Index >
bool
tnlDirichletBoundaryConditions< Mesh, Function, MeshEntitiesDimension, Real, Index >::
setup( const Config::ParameterContainer& parameters,
       const String& prefix )
{
   return this->function.setup( parameters );
}

template< typename Mesh,
          typename Function,
          int MeshEntitiesDimension,
          typename Real,
          typename Index >
void
tnlDirichletBoundaryConditions< Mesh, Function, MeshEntitiesDimension, Real, Index >::
setFunction( const Function& function )
{
   this->function = function;
}

template< typename Mesh,
          typename Function,
          int MeshEntitiesDimension,
          typename Real,
          typename Index >
Function&
tnlDirichletBoundaryConditions< Mesh, Function, MeshEntitiesDimension, Real, Index >::
getFunction()
{
   return this->function;
}

template< typename Mesh,
          typename Function,
          int MeshEntitiesDimension,
          typename Real,
          typename Index >
const Function&
tnlDirichletBoundaryConditions< Mesh, Function, MeshEntitiesDimension, Real, Index >::
getFunction() const
{
   return *this->function;
}


template< typename Mesh,
          typename Function,
          int MeshEntitiesDimension,
          typename Real,
          typename Index >
template< typename EntityType,
          typename MeshFunction >
__cuda_callable__
const Real
tnlDirichletBoundaryConditions< Mesh, Function, MeshEntitiesDimension, Real, Index >::
operator()( const MeshFunction& u,
            const EntityType& entity,            
            const RealType& time ) const
{
   //static_assert( EntityType::getDimensions() == MeshEntitiesDimension, "Wrong mesh entity dimensions." );
   return tnlFunctionAdapter< MeshType, Function >::template getValue( this->function, entity, time );
}

template< typename Mesh,
          typename Function,
          int MeshEntitiesDimension,
          typename Real,
          typename Index >
   template< typename EntityType >
__cuda_callable__
Index
tnlDirichletBoundaryConditions< Mesh, Function, MeshEntitiesDimension, Real, Index >::
getLinearSystemRowLength( const MeshType& mesh,
                          const IndexType& index,
                          const EntityType& entity ) const
{
   return 1;
}

template< typename Mesh,
          typename Function,
          int MeshEntitiesDimension,
          typename Real,
          typename Index >
   template< typename Matrix,
             typename EntityType,
             typename MeshFunction >
__cuda_callable__
void
tnlDirichletBoundaryConditions< Mesh, Function, MeshEntitiesDimension, Real, Index >::
updateLinearSystem( const RealType& time,
                    const MeshType& mesh,
                    const IndexType& index,
                    const EntityType& entity,
                    const MeshFunction& u,
                    DofVectorType& b,
                    Matrix& matrix ) const
{
   typename Matrix::MatrixRow matrixRow = matrix.getRow( index );
   matrixRow.setElement( 0, index, 1.0 );
   b[ index ] = tnlFunctionAdapter< MeshType, Function >::getValue( this->function, entity, time );
}

#endif /* TNLDIRICHLETBOUNDARYCONDITIONS_IMPL_H_ */
