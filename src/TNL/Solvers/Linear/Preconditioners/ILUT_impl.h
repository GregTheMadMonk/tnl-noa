/***************************************************************************
                          ILUT_impl.h  -  description
                             -------------------
    begin                : Aug 31, 2018
    copyright            : (C) 2018 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

// Implemented by: Jakub Klinkovsky

#pragma once

#include <vector>

#include "ILUT.h"
#include "TriangularSolve.h"

#include <TNL/Timer.h>

namespace TNL {
namespace Solvers {
namespace Linear {
namespace Preconditioners {

template< typename Matrix, typename Real, typename Index >
bool
ILUT_impl< Matrix, Real, Devices::Host, Index >::
setup( const Config::ParameterContainer& parameters,
       const String& prefix )
{
   p = parameters.getParameter< int >( "ilut-p" );
   tau = parameters.getParameter< double >( "ilut-threshold" );
   return true;
}

template< typename Matrix, typename Real, typename Index >
void
ILUT_impl< Matrix, Real, Devices::Host, Index >::
update( const MatrixPointer& matrixPointer )
{
   TNL_ASSERT_GT( matrixPointer->getRows(), 0, "empty matrix" );
   TNL_ASSERT_EQ( matrixPointer->getRows(), matrixPointer->getColumns(), "matrix must be square" );

   const auto& localMatrix = Traits< Matrix >::getLocalMatrix( *matrixPointer );
   const IndexType N = localMatrix.getRows();
   const IndexType minColumn = getMinColumn( *matrixPointer );

   L.setDimensions( N, N );
   U.setDimensions( N, N );

   Timer timer_total, timer_rowlengths, timer_copy_into_w, timer_k_loop, timer_dropping, timer_copy_into_LU;

   timer_total.start();

   // compute row lengths
   timer_rowlengths.start();
   typename decltype(L)::CompressedRowLengthsVector L_rowLengths;
   typename decltype(U)::CompressedRowLengthsVector U_rowLengths;
   L_rowLengths.setSize( N );
   U_rowLengths.setSize( N );
   for( IndexType i = 0; i < N; i++ ) {
      const auto row = localMatrix.getRow( i );
      const auto max_length = localMatrix.getRowLength( i );
      IndexType L_entries = 0;
      IndexType U_entries = 0;
      for( IndexType j = 0; j < max_length; j++ ) {
         const auto column = row.getElementColumn( j );
         if( column < minColumn )
            continue;
         if( column < i + minColumn )
            L_entries++;
         else if( column < N + minColumn )
            U_entries++;
         else
            break;
      }
      // store p additional entries in each factor
      L_rowLengths[ i ] = L_entries + p;
      U_rowLengths[ N - 1 - i ] = U_entries + p;
   }
   L.setCompressedRowLengths( L_rowLengths );
   U.setCompressedRowLengths( U_rowLengths );
   timer_rowlengths.stop();

   // intermediate full vector for the i-th row of A
   VectorType w;
   w.setSize( N );
   w.setValue( 0.0 );

   // intermediate vectors for sorting and keeping only the largest values
//   using Pair = std::pair< IndexType, RealType >;
   struct Triplet {
      IndexType column;
      RealType value;
      RealType abs_value;
      Triplet(IndexType column, RealType value, RealType abs_value) : column(column), value(value), abs_value(abs_value) {}
   };
   auto cmp_abs_value = []( const Triplet& a, const Triplet& b ){ return a.abs_value < b.abs_value; };
   std::vector< Triplet > heap_L, heap_U;
   auto cmp_column = []( const Triplet& a, const Triplet& b ){ return a.column < b.column; };
   std::vector< Triplet > values_L, values_U;

//   std::cout << "N = " << N << std::endl;

   // Incomplete LU factorization with threshold
   // (see Saad - Iterative methods for sparse linear systems, section 10.4)
   for( IndexType i = 0; i < N; i++ ) {
      const auto max_length = localMatrix.getRowLength( i );
      const auto A_i = localMatrix.getRow( i );

      RealType A_i_norm = 0.0;

      // copy A_i into the full vector w
      timer_copy_into_w.start();
      for( IndexType c_j = 0; c_j < max_length; c_j++ ) {
         auto j = A_i.getElementColumn( c_j );
         if( minColumn > 0 ) {
            // skip non-local elements
            if( j < minColumn ) continue;
            j -= minColumn;
         }
         // handle ellpack dummy entries
         if( j >= N ) break;
         w[ j ] = A_i.getElementValue( c_j );

         // running computation of norm
         A_i_norm += w[ j ] * w[ j ];
      }
      timer_copy_into_w.stop();

      // compute relative tolerance
      A_i_norm = std::sqrt( A_i_norm );
      const RealType tau_i = tau * A_i_norm;

      // loop for k = 0, ..., i - 1; but only over the non-zero entries of w
      timer_k_loop.start();
      for( IndexType k = 0; k < i; k++ ) {
         RealType w_k = w[ k ];
         if( w_k == 0.0 )
            continue;

         w_k /= localMatrix.getElementFast( k, k + minColumn );

         // apply dropping rule to w_k
         if( std::abs( w_k ) < tau_i )
            w_k = 0.0;

         w[ k ] = w_k;

         if( w_k != 0.0 ) {
            // w := w - w_k * U_k
            const auto U_k = U.getRow( N - 1 - k );
            // loop for j = 0, ..., N-1; but only over the non-zero entries
            for( Index c_j = 0; c_j < U_rowLengths[ N - 1 - k ]; c_j++ ) {
               const auto j = U_k.getElementColumn( c_j );
               // skip dropped entries
               if( j >= N ) break;
               w[ j ] -= w_k * U_k.getElementValue( c_j );
            }
         }
      }
      timer_k_loop.stop();

      // apply dropping rule to the row w
      // (we drop all values under threshold and keep nl(i) + p largest values in L
      // and nu(i) + p largest values in U; see Saad (2003) for reference)
      // TODO: refactoring!!! (use the quick-split strategy, constructing the heap is not necessary)
      timer_dropping.start();
      for( IndexType j = 0; j < N; j++ ) {
         const RealType w_j_abs = std::abs( w[ j ] );
         // ignore small values
         if( w_j_abs < tau_i )
            continue;
         // push into the heaps for L or U
         if( j < i ) {
            heap_L.push_back( Triplet( j, w[ j ], w_j_abs ) );
            std::push_heap( heap_L.begin(), heap_L.end(), cmp_abs_value );
         }
         else {
            heap_U.push_back( Triplet( j, w[ j ], w_j_abs ) );
            std::push_heap( heap_U.begin(), heap_U.end(), cmp_abs_value );
         }
      }
      // extract values for L and U
      for( IndexType c_j = 0; c_j < L_rowLengths[ i ] && c_j < heap_L.size(); c_j++ ) {
         // move the largest to the end
         std::pop_heap( heap_L.begin(), heap_L.end(), cmp_abs_value );
         // move the triplet from one vector into another
         const auto largest = heap_L.back();
         heap_L.pop_back();
         values_L.push_back( largest );
      }
      for( IndexType c_j = 0; c_j < U_rowLengths[ N - 1 - i ] && c_j < heap_U.size(); c_j++ ) {
         // move the largest to the end
         std::pop_heap( heap_U.begin(), heap_U.end(), cmp_abs_value );
         // move the triplet from one vector into another
         const auto largest = heap_U.back();
         heap_U.pop_back();
         values_U.push_back( largest );
      }
      // sort by column index to make it insertable into the sparse matrix
      std::sort( values_L.begin(), values_L.end(), cmp_column );
      std::sort( values_U.begin(), values_U.end(), cmp_column );
      timer_dropping.stop();

//      std::cout << "i = " << i << ", L_rowLengths[ i ] = " << L_rowLengths[ i ] << ", U_rowLengths[ i ] = " << U_rowLengths[ N - 1 - i ] << std::endl;

      timer_copy_into_LU.start();

      // the row L_i might be empty
      if( values_L.size() ) {
         // L_ij = w_j for j = 0, ..., i - 1
         auto L_i = L.getRow( i );
         for( IndexType c_j = 0; c_j < values_L.size(); c_j++ ) {
            const auto j = values_L[ c_j ].column;
//            std::cout << "c_j = " << c_j << ", j = " << j << std::endl;
            L_i.setElement( c_j, j, values_L[ c_j ].value );
         }
      }

      // U_ij = w_j for j = i, ..., N - 1
      auto U_i = U.getRow( N - 1 - i );
      for( IndexType c_j = 0; c_j < values_U.size(); c_j++ ) {
         const auto j = values_U[ c_j ].column;
//         std::cout << "c_j = " << c_j << ", j = " << j << std::endl;
         U_i.setElement( c_j, j, values_U[ c_j ].value );
      }

      timer_copy_into_LU.stop();

      // reset w
      w.setValue( 0.0 );

      heap_L.clear();
      heap_U.clear();
      values_L.clear();
      values_U.clear();
   }

   timer_total.stop();

   std::cout << "ILUT::update statistics:\n";
   std::cout << "\ttimer_total:        " << timer_total.getRealTime()         << " s\n";
   std::cout << "\ttimer_rowlengths:   " << timer_rowlengths.getRealTime()    << " s\n";
   std::cout << "\ttimer_copy_into_w:  " << timer_copy_into_w.getRealTime()   << " s\n";
   std::cout << "\ttimer_k_loop:       " << timer_k_loop.getRealTime()        << " s\n";
   std::cout << "\ttimer_dropping:     " << timer_dropping.getRealTime()      << " s\n";
   std::cout << "\ttimer_copy_into_LU: " << timer_copy_into_LU.getRealTime()  << " s\n";
   std::cout << std::flush;
}

template< typename Matrix, typename Real, typename Index >
void
ILUT_impl< Matrix, Real, Devices::Host, Index >::
solve( ConstVectorViewType _b, VectorViewType _x ) const
{
   const auto b = Traits< Matrix >::getLocalVectorView( _b );
   auto x = Traits< Matrix >::getLocalVectorView( _x );

   // Step 1: solve y from Ly = b
   triangularSolveLower< false >( L, x, b );

   // Step 2: solve x from Ux = y
   triangularSolveUpper< true, false >( U, x, x );
}

} // namespace Preconditioners
} // namespace Linear
} // namespace Solvers
} // namespace TNL
