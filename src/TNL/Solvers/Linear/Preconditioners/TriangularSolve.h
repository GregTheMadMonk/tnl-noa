/***************************************************************************
                          TriangularSolve.h  -  description
                             -------------------
    begin                : Sep 21, 2018
    copyright            : (C) 2018 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

// Implemented by: Jakub Klinkovsky

#pragma once

namespace TNL {
namespace Solvers {
namespace Linear {
namespace Preconditioners {

/*
 * Solves `x` from `Lx = b`, where L is a square lower triangular matrix with
 * implicit ones on the diagonal.
 *
 * Note that the solution can be updated in-place if `x` and `b` are passed
 * as the same vector.
 *
 * If `fullStorage` is true, then it is assumed that all non-zero entries of the
 * matrix are valid. Otherwise an explicit check is performed to detect padding
 * zeros. This is useful for Ellpack-based formats or if more space than
 * necessary was explicitly allocated.
 */
template< bool fullStorage = true, typename Matrix, typename Vector1, typename Vector2 >
void triangularSolveLower( const Matrix& L, Vector1& x, const Vector2& b )
{
   TNL_ASSERT_EQ( b.getSize(), L.getRows(), "wrong size of the right hand side" );
   TNL_ASSERT_EQ( x.getSize(), L.getRows(), "wrong size of the solution vector" );

   using RealType = typename Vector1::RealType;
   using IndexType = typename Vector1::IndexType;

   const IndexType N = x.getSize();

   for( IndexType i = 0; i < N; i++ ) {
      RealType x_i = b[ i ];

      const auto L_entries = L.getRowCapacity( i );

      // this condition is to avoid segfaults on empty L.getRow( i )
      if( L_entries > 0 ) {
         const auto L_i = L.getRow( i );

         // loop for j = 0, ..., i - 1; but only over the non-zero entries
         for( IndexType c_j = 0; c_j < L_entries; c_j++ ) {
            const auto j = L_i.getColumnIndex( c_j );
            // skip padding zeros
            if( fullStorage == false && j >= N )
               break;
            x_i -= L_i.getValue( c_j ) * x[ j ];
         }
      }

      x[ i ] = x_i;
   }
}

/*
 * Solves `x` from `Ux = b`, where U is a square upper triangular matrix.
 *
 * Note that the solution can be updated in-place if `x` and `b` are passed
 * as the same vector.
 *
 * If `reversedRows` is true, the rows of `U` are indexed in reverse order.
 *
 * If `fullStorage` is true, then it is assumed that all non-zero entries of the
 * matrix are valid. Otherwise an explicit check is performed to detect padding
 * zeros. This is useful for Ellpack-based formats or if more space than
 * necessary was explicitly allocated.
 */
template< bool reversedRows = false, bool fullStorage = true,
          typename Matrix, typename Vector1, typename Vector2 >
void triangularSolveUpper( const Matrix& U, Vector1& x, const Vector2& b )
{
   TNL_ASSERT_EQ( b.getSize(), U.getRows(), "wrong size of the right hand side" );
   TNL_ASSERT_EQ( x.getSize(), U.getRows(), "wrong size of the solution vector" );

   using RealType = typename Vector1::RealType;
   using IndexType = typename Vector1::IndexType;

   const IndexType N = x.getSize();

   for( IndexType i = N - 1; i >= 0; i-- ) {
      RealType x_i = b[ i ];

      const IndexType U_idx = (reversedRows) ? N - 1 - i : i;

      const auto U_entries = U.getRowCapacity( U_idx );
      const auto U_i = U.getRow( U_idx );

      const auto U_ii = U_i.getValue( 0 );

      // loop for j = i+1, ..., N-1; but only over the non-zero entries
      for( IndexType c_j = 1; c_j < U_entries ; c_j++ ) {
         const auto j = U_i.getColumnIndex( c_j );
         // skip padding zeros
         if( fullStorage == false && j >= N )
            break;
         x_i -= U_i.getValue( c_j ) * x[ j ];
      }

      x[ i ] = x_i / U_ii;
   }
}

} // namespace Preconditioners
} // namespace Linear
} // namespace Solvers
} // namespace TNL
