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

template< int Dimensions,
          typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Vector,
          typename Real,
          typename Index >
void
tnlDirichletBoundaryConditions< tnlGrid< Dimensions, MeshReal, Device, MeshIndex >, Vector, Real, Index >::
configSetup( tnlConfigDescription& config,
             const tnlString& prefix )
{
   config.addEntry     < tnlString >( prefix + "file", "Data for the boundary conditions." );
}

template< int Dimensions,
          typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Vector,
          typename Real,
          typename Index >
bool
tnlDirichletBoundaryConditions< tnlGrid< Dimensions, MeshReal, Device, MeshIndex >, Vector, Real, Index >::
setup( const tnlParameterContainer& parameters,
       const tnlString& prefix )
{
   if( parameters.checkParameter( prefix + "file" ) )
   {
      tnlString fileName = parameters.getParameter< tnlString >( prefix + "file" );
      if( ! this->vector.load( fileName ) )
         return false;
   }
   return true;
}

template< int Dimensions,
          typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Vector,
          typename Real,
          typename Index >
Vector&
tnlDirichletBoundaryConditions< tnlGrid< Dimensions, MeshReal, Device, MeshIndex >, Vector, Real, Index >::
getVector()
{
   return this->vector;
}

template< int Dimensions,
          typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Vector,
          typename Real,
          typename Index >
const Vector&
tnlDirichletBoundaryConditions< tnlGrid< Dimensions, MeshReal, Device, MeshIndex >, Vector, Real, Index >::
getVector() const
{
   return this->vector;
}


template< int Dimensions,
          typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Vector,
          typename Real,
          typename Index >
__cuda_callable__
void
tnlDirichletBoundaryConditions< tnlGrid< Dimensions, MeshReal, Device, MeshIndex >, Vector, Real, Index >::
setBoundaryConditions( const RealType& time,
                       const MeshType& mesh,
                       const IndexType index,
                       const CoordinatesType& coordinates,
                       DofVectorType& u,
                       DofVectorType& fu ) const
{
   fu[ index ] = 0;
   u[ index ] = this->vector[ index ];
}

template< int Dimensions,
          typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Vector,
          typename Real,
          typename Index >
__cuda_callable__
Index
tnlDirichletBoundaryConditions< tnlGrid< Dimensions, MeshReal, Device, MeshIndex >, Vector, Real, Index >::
getLinearSystemRowLength( const MeshType& mesh,
                          const IndexType& index,
                          const CoordinatesType& coordinates ) const
{
   return 1;
}

template< int Dimensions,
          typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Vector,
          typename Real,
          typename Index >
   template< typename Matrix >
__cuda_callable__
void
tnlDirichletBoundaryConditions< tnlGrid< Dimensions, MeshReal, Device, MeshIndex >, Vector, Real, Index >::
updateLinearSystem( const RealType& time,
                    const MeshType& mesh,
                    const IndexType& index,
                    const CoordinatesType& coordinates,
                    DofVectorType& u,
                    DofVectorType& b,
                    Matrix& matrix ) const
{
   typename Matrix::MatrixRow matrixRow = matrix.getRow( index );
   matrixRow.setElement( 0, index, 1.0 );
   b[ index ] = this->vector[ index ];
}



#endif /* TNLDIRICHLETBOUNDARYCONDITIONS_IMPL_H_ */
