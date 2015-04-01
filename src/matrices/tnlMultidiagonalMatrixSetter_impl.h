/***************************************************************************
                          tnlMultidiagonalMatrixSetter_impl.h  -  description
                             -------------------
    begin                : Jan 2, 2015
    copyright            : (C) 2015 by oberhuber
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

#ifndef TNLMULTIDIAGONALMATRIXSETTER_IMPL_H_
#define TNLMULTIDIAGONALMATRIXSETTER_IMPL_H_

template< typename MeshReal,
          typename Device,
          typename MeshIndex >
   template< typename Real,
             typename Index >
bool
tnlMultidiagonalMatrixSetter< tnlGrid< 1, MeshReal, Device, MeshIndex > >::
setupMatrix( const MeshType& mesh,
             tnlMultidiagonalMatrix< Real, Device, Index >& matrix,
             int stencilSize,
             bool crossStencil )
{
   const Index dofs = mesh.getNumberOfCells();
   matrix.setDimensions( dofs, dofs );
   CoordinatesType centerCell( stencilSize );
   tnlVector< Index, Device, Index > diagonals;
   if( ! diagonals.setSize( 3 ) )
      return false;
   Index centerCellIndex = mesh.getCellIndex( CoordinatesType( stencilSize ) );
   diagonals.setElement( 0, mesh.getCellIndex( CoordinatesType( stencilSize - 1 ) ) - centerCellIndex );
   diagonals.setElement( 1, 0 );
   diagonals.setElement( 2, mesh.getCellIndex( CoordinatesType( stencilSize + 1 ) ) - centerCellIndex );
   //cout << "Setting the multidiagonal matrix offsets to: " << diagonals << endl;
   return matrix.setDiagonals( diagonals );
}

template< typename MeshReal,
          typename Device,
          typename MeshIndex >
   template< typename Real,
             typename Index >
bool
tnlMultidiagonalMatrixSetter< tnlGrid< 2, MeshReal, Device, MeshIndex > >::
setupMatrix( const MeshType& mesh,
             tnlMultidiagonalMatrix< Real, Device, Index >& matrix,
             int stencilSize,
             bool crossStencil )
{
   const Index dofs = mesh.getNumberOfCells();
   matrix.setDimensions( dofs, dofs );
   CoordinatesType centerCell( stencilSize );
   tnlVector< Index, Device, Index > diagonals;
   if( ! diagonals.setSize( 5 ) )
      return false;
   Index centerCellIndex = mesh.getCellIndex( CoordinatesType( stencilSize, stencilSize ) );
   diagonals.setElement( 0, mesh.getCellIndex( CoordinatesType( stencilSize, stencilSize - 1 ) ) - centerCellIndex );
   diagonals.setElement( 1, mesh.getCellIndex( CoordinatesType( stencilSize - 1, stencilSize ) ) - centerCellIndex );
   diagonals.setElement( 2, 0 );
   diagonals.setElement( 3, mesh.getCellIndex( CoordinatesType( stencilSize + 1, stencilSize ) ) - centerCellIndex );
   diagonals.setElement( 4, mesh.getCellIndex( CoordinatesType( stencilSize, stencilSize + 1 ) ) - centerCellIndex );
   //cout << "Setting the multidiagonal matrix offsets to: " << diagonals << endl;
   return matrix.setDiagonals( diagonals );
}

template< typename MeshReal,
          typename Device,
          typename MeshIndex >
   template< typename Real,
             typename Index >
bool
tnlMultidiagonalMatrixSetter< tnlGrid< 3, MeshReal, Device, MeshIndex > >::
setupMatrix( const MeshType& mesh,
             tnlMultidiagonalMatrix< Real, Device, Index >& matrix,
             int stencilSize,
             bool crossStencil )
{
   const Index dofs = mesh.getNumberOfCells();
   matrix.setDimensions( dofs, dofs );
   CoordinatesType centerCell( stencilSize );
   tnlVector< Index, Device, Index > diagonals;
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
   //cout << "Setting the multidiagonal matrix offsets to: " << diagonals << endl;
   return matrix.setDiagonals( diagonals );
}

#endif /* TNLMULTIDIAGONALMATRIXSETTER_IMPL_H_ */
