/***************************************************************************
                          MultidiagonalMatrixSetter_impl.h  -  description
                             -------------------
    begin                : Jan 2, 2015
    copyright            : (C) 2015 by oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

namespace TNL {
namespace Matrices {   

template< typename MeshReal,
          typename Device,
          typename MeshIndex >
   template< typename Real,
             typename Index >
bool
MultidiagonalMatrixSetter< Meshes::Grid< 1, MeshReal, Device, MeshIndex > >::
setupMatrix( const MeshType& mesh,
             MultidiagonalMatrix< Real, Device, Index >& matrix,
             int stencilSize,
             bool crossStencil )
{
   const Index dofs = mesh.template getEntitiesCount< typename MeshType::Cell >();
   matrix.setDimensions( dofs, dofs );
   CoordinatesType centerCell( stencilSize );
   Vectors::Vector< Index, Device, Index > diagonals;
   if( ! diagonals.setSize( 3 ) )
      return false;
   Index centerCellIndex = mesh.getCellIndex( CoordinatesType( stencilSize ) );
   diagonals.setElement( 0, mesh.getCellIndex( CoordinatesType( stencilSize - 1 ) ) - centerCellIndex );
   diagonals.setElement( 1, 0 );
   diagonals.setElement( 2, mesh.getCellIndex( CoordinatesType( stencilSize + 1 ) ) - centerCellIndex );
   //cout << "Setting the multidiagonal matrix offsets to: " << diagonals << std::endl;
   return matrix.setDiagonals( diagonals );
}

template< typename MeshReal,
          typename Device,
          typename MeshIndex >
   template< typename Real,
             typename Index >
bool
MultidiagonalMatrixSetter< Meshes::Grid< 2, MeshReal, Device, MeshIndex > >::
setupMatrix( const MeshType& mesh,
             MultidiagonalMatrix< Real, Device, Index >& matrix,
             int stencilSize,
             bool crossStencil )
{
   const Index dofs = mesh.template getEntitiesCount< typename MeshType::Cell >();
   matrix.setDimensions( dofs, dofs );
   CoordinatesType centerCell( stencilSize );
   Vectors::Vector< Index, Device, Index > diagonals;
   if( ! diagonals.setSize( 5 ) )
      return false;
   Index centerCellIndex = mesh.getCellIndex( CoordinatesType( stencilSize, stencilSize ) );
   diagonals.setElement( 0, mesh.getCellIndex( CoordinatesType( stencilSize, stencilSize - 1 ) ) - centerCellIndex );
   diagonals.setElement( 1, mesh.getCellIndex( CoordinatesType( stencilSize - 1, stencilSize ) ) - centerCellIndex );
   diagonals.setElement( 2, 0 );
   diagonals.setElement( 3, mesh.getCellIndex( CoordinatesType( stencilSize + 1, stencilSize ) ) - centerCellIndex );
   diagonals.setElement( 4, mesh.getCellIndex( CoordinatesType( stencilSize, stencilSize + 1 ) ) - centerCellIndex );
   //cout << "Setting the multidiagonal matrix offsets to: " << diagonals << std::endl;
   return matrix.setDiagonals( diagonals );
}

template< typename MeshReal,
          typename Device,
          typename MeshIndex >
   template< typename Real,
             typename Index >
bool
MultidiagonalMatrixSetter< Meshes::Grid< 3, MeshReal, Device, MeshIndex > >::
setupMatrix( const MeshType& mesh,
             MultidiagonalMatrix< Real, Device, Index >& matrix,
             int stencilSize,
             bool crossStencil )
{
   const Index dofs = mesh.template getEntitiesCount< typename MeshType::Cell >();
   matrix.setDimensions( dofs, dofs );
   CoordinatesType centerCell( stencilSize );
   Vectors::Vector< Index, Device, Index > diagonals;
   if( ! diagonals.setSize( 7 ) )
      return false;
   Index centerCellIndex = mesh.getCellIndex( CoordinatesType( stencilSize, stencilSize, stencilSize ) );
   diagonals.setElement( 0, mesh.getCellIndex( CoordinatesType( stencilSize, stencilSize, stencilSize - 1 ) ) - centerCellIndex );
   diagonals.setElement( 1, mesh.getCellIndex( CoordinatesType( stencilSize, stencilSize - 1, stencilSize ) ) - centerCellIndex );
   diagonals.setElement( 2, mesh.getCellIndex( CoordinatesType( stencilSize - 1, stencilSize, stencilSize ) ) - centerCellIndex );
   diagonals.setElement( 3, 0 );
   diagonals.setElement( 4, mesh.getCellIndex( CoordinatesType( stencilSize + 1, stencilSize, stencilSize ) ) - centerCellIndex );
   diagonals.setElement( 5, mesh.getCellIndex( CoordinatesType( stencilSize, stencilSize + 1, stencilSize ) ) - centerCellIndex );
   diagonals.setElement( 6, mesh.getCellIndex( CoordinatesType( stencilSize, stencilSize, stencilSize + 1 ) ) - centerCellIndex );
   //cout << "Setting the multidiagonal matrix offsets to: " << diagonals << std::endl;
   return matrix.setDiagonals( diagonals );
}

} // namespace Matrices
} // namespace TNL
