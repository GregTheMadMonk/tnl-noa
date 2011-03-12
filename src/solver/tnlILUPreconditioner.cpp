/***************************************************************************
                          tnlILUPreconditioner.cpp  -  description
                             -------------------
    begin                : 2007/02/01
    copyright            : (C) 2007 by Tomas Oberhuber
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

#include <assert.h>
#include <math.h>
#include "tnlILUPreconditioner.h"
#include "mfuncs.h"

#ifdef ILU_DEBUG
#include <iomanip>
#include <fstream>
//--------------------------------------------------------------------------
void WriteFullMatrix( ostream& o_str, const m_real* M, const m_int N )
{
   m_int i, j;
   for( i = 0; i < N; i ++ )
   {
      for( j = 0; j < N; j ++ )
      {
         const m_real& v = M[ i * N + j ];
         if( v == 0.0 ) o_str << setw( 10 ) << ".";
         else o_str << setprecision( 6 ) << setw( 10 ) << v;
      }
      o_str << endl;
   }
}
#endif
//--------------------------------------------------------------------------
tnlILUPreconditioner :: tnlILUPreconditioner( const m_int _size, 
                                          const m_int initial_size,
                                          const m_int segment_size,
                                          const m_int init_row_elements )
: size( _size )
{
#ifdef ILU_DEBUG
   full_M = new m_real[ size * size ];
   ilu_check = new m_real[ size * size ];
#endif
   y = new m_real[ size ];
   M = new tnlCSRMatrix( size, initial_size, segment_size, init_row_elements );
}
//--------------------------------------------------------------------------  
tnlILUPreconditioner :: ~tnlILUPreconditioner()
{
   if( M ) delete M;
   if( y ) delete[] y;
#ifdef ILU_DEBUG
   if( full_M ) delete[] full_M;
   if( ilu_check ) delete[] ilu_check;
#endif
}
//--------------------------------------------------------------------------  
const tnlCSRMatrix* tnlILUPreconditioner :: getVector() const
{
   return M;
}
//--------------------------------------------------------------------------  
m_int tnlILUPreconditioner :: Init( const tnlCSRMatrix& A, const m_real& threshold )
{
   dbgFunctionName( "tnlILUPreconditioner", "Init" );
   assert( A. Size() == M -> Size() );
   int non_zero_elements( 0 );
#ifdef CSR_MATRIX_TUNING
   M -> ResetStatistics();
#endif
   m_int i, j, k;
   const tnlCSRMatrixElement *A_data;
   const tnlCSRMatrixRowInfo *A_rows_info;
   A. getVector( A_data, A_rows_info );
   
#ifdef ILU_DEBUG
   // 1. Copy data of A to full_M
   bzero( full_M, size * size * sizeof( m_real ) );
   for( i = 0; i < size; i ++ )
      for( j = A_rows_info[ i ]. first; j < A_rows_info[ i ]. last; j ++ )
      {
         m_int col = A_data[ j ]. column;
         if( col != -1 )
            full_M[ i * size + col ] = A_data[ j ]. value;
      }
#endif

   M -> Reset();
   tnlCSRMatrixElement* M_data;
   const tnlCSRMatrixRowInfo *M_rows_info;
   // 2. Processing IKJ version of ILU factorisation
   // For i = 0, ... , N
   for( i = 0; i < size; i ++ )
   {
      // y = a_i*  - i-th row
      bzero( y, size * sizeof( m_real ) );
      m_real y_norm( 0.0 );
      const m_int a_row_beg = A_rows_info[ i ]. first;
      const m_int a_row_end = A_rows_info[ i ]. last;
      for( k = a_row_beg; k < a_row_end; k ++ )
      {
         const m_int col = A_data[ k ]. column;
         const m_real& a_ik = A_data[ k ]. value;
         if( col == -1 ) break;
         y[ col ] = a_ik;
         y_norm += a_ik * a_ik;
      }
      // get 2-norm of the i-th row for the dropping rule on y_k
      y_norm = sqrt( y_norm );
      const m_real threshold_i = y_norm * threshold;
      M -> Data( M_data, M_rows_info );
      for( k = 0; k < i; k ++ )
      {
         const m_real& m_kk = ( *M )( k, k );
         if( m_kk != 0.0 )
            y[ k ] /= m_kk;
         if( fabs( y[ k ] ) < threshold_i ) y[ k ] = 0.0;
         if( y[ k ] != 0.0 )
         {
            const m_int row_beg = M_rows_info[ k ]. diagonal + 1;
            const m_int row_end = M_rows_info[ k ]. last;
            m_int col;
            for( j = row_beg;
                 j < row_end && ( col = M_data[ j ]. column ) != -1;
                 j ++ )
               y[ col ] -= y[ k ] * M_data[ j ]. value;
         }
      }
      for( j = 0; j < size; j ++ )
      {
         if( i == j || fabs( y[ j ] ) > threshold_i )
         {
             M -> Set( i, j , y[ j ] );
             non_zero_elements ++;
         }
         if( i == j && ( *M )( i, i ) == 0.0 )
         {
            cerr << endl << "Zero pivot appears at line " << i << "!" << endl;
            M -> Set( i, i, 1.0 );
            abort();
            //return 0;
         }
      }
     cout << "Computing ILUT ... " << 100.0 * ( m_real ) ( i + 1 ) / ( m_real ) ( size ) << "%    \r" << flush; 
      dbgExpr( i );
      //dbgExpr( * M );
   }
   cout << non_zero_elements / size << " elems. per line in average " <<  endl;
   //dbgExpr( * M );

#ifdef ILU_DEBUG
   fstream file;
   /*file. open( "orig_full_M2", ios :: out );
   WriteFullMatrix( file, full_M, size );
   file. close();*/
   
   /*file. open( "M", ios :: out );
   file << *M << endl;
   file. close();*/

   for( i = 0; i < size; i ++ )
   {
cout << "LU: " << ( m_real ) i / ( m_real ) size * 100.0 << " % done   \r" << flush;
      // y = a_i*  - i-th row
      bzero( y, size * sizeof( m_real ) );
      m_real y_norm( 0.0 );
      for( k = 0; k < size; k ++ )
      {
         const m_real& m_ik = full_M[ i * size + k ];
         y[ k ] = m_ik;
         y_norm += m_ik * m_ik;
         //cout << i << ", " << k << " <- " << y[ k ] << endl;
      }
      // get 2-norm of the i-th row for the dropping rule on y_k
      y_norm = sqrt( y_norm );
      const m_real threshold_i = y_norm * threshold;
      for( k = 0; k < i; k ++ )
      {
         const m_real& m_kk = full_M[ k * size + k ];
         if( m_kk != 0.0 )
            y[ k ] /= m_kk;
         if( fabs( y[ k ] ) < threshold_i )
            y[ k ] = 0.0;
         if( y[ k ] != 0.0 )
            for( j = k + 1; j < size; j ++ )
               y[ j ] -= y[ k ] * full_M[ k * size + j ];
      }
      for( j = 0; j < size; j ++ )
      {
         if( i == j || fabs( y[ j ] ) > threshold_i )
            full_M[ i * size + j ] = y[ j ];
         else
            full_M[ i * size + j  ] = 0.0;
      }
      if( full_M[ i * size + i ] == 0.0 )
         cerr << endl << "FULL MATRIX ILU CHECK: Zero pivot appears at line " << i << "!" << endl;
   }
   /*file. open( "full_M", ios :: out );
   WriteFullMatrix( file, full_M, size );
   file. close();*/

   cout << endl;
   cout << "Checking if M == full M..." << endl;
   for( i = 0; i < size; i ++ )
      for( j = 0; j < size; j ++ )
         if( full_M[ i * size + j ] != ( *M )( i, j ) )
         {
            cerr << "M != full M at " << i << ", " << j << " : " <<
            fabs( full_M[ i * size + j ] - ( *M )( i, j ) ) << endl;
            break;
         }
   
   /*cout << "Checking  if LU = A" << endl;
   // check whether L U == A
   for( i = 0; i < size; i ++ )
      for( j = 0; j < size; j ++ )
      {
         cout << i << " : " << j << "     \r" << flush;
         m_int l = Min( i, j );
         m_real sum( 0.0 );
         for( k = 0; k <= l; k ++ )
            if( k < i )
               sum += full_M[ i * size + k ] * full_M[ k * size + j ];
            else
               sum += full_M[ i * size + j ];  // L has 1 on diagonal
         if( fabs( sum ) < 1.0e-15 ) sum = 0.0;
         ilu_check[ i * size + j ] = sum;
      }
   cout << endl;
   m_real ilu_diff( 0.0 );
   for( i = 0; i < size; i ++ )
      for( j = 0; j < size; j ++ )
         ilu_diff = Max( ilu_diff, fabs( ilu_check[ i * size +j ] - A( i, j ) ) );
   cout << "ILU Max. error is " << ilu_diff << endl;*/
#endif

#ifdef CSR_MATRIX_TUNING
   M -> PrintStatistics();
#endif
   return 1;  
   
     
   /*
   // For i = 1 ... n - 1
   for( i = 1; i < size; i ++ )
   {
      // For k = 0 ... i - 1 and for ( i, k ) non-zero
      //  = take non-zero entries at the i-th row before the diagonal entry
      m_int i_th_row_beg = M_row_ind[ i ];
      m_int i_th_row_end = M_row_ind[ i + 1 ];
      m_int k_ind;
      for( k_ind = i_th_row_beg; k_ind < i_th_row_end; k_ind ++ )
      {
         k = M_data[ k_ind ]. column;
         if( k == -1 ) break;
         // a_ik := a_ik / a_kk
         M_data[ k_ind ]. value /= M_data[ M_diag_ind[ k ] ]. value;
         const m_real& a_ik = M_data[ k_ind ]. value;

         // For j = k + 1 ... n - 1 and for ( i, j ) non-zero
         // = take non-zero entries at the i-th row behind the diagonal entry
         m_int j_ind;
         for( j_ind = k_ind + 1; j_ind < i_th_row_end; j_ind ++ )
         {
            j = M_data[ j_ind ]. column;
            if( j == -1 ) break;
            // a_ij :=  a_ij - a_ik * a_kj
            M_data[ j_ind ]. value -= a_ik * ( *M )( k, j );
         }
      }
   }*/

}
//--------------------------------------------------------------------------     
void tnlILUPreconditioner :: Solve( const m_real* b, m_real* x ) const
{
   dbgFunctionName( "tnlILUPreconditioner", "Solve" );
   const m_int size = M -> Size();
   const tnlCSRMatrixElement* M_data;
   const tnlCSRMatrixRowInfo *M_rows_info;
   M -> Data( M_data, M_rows_info );
   m_int i, j;
   
   dbgCout( "Solving Ly = b" );
   // L is unit lower triangular
   for( i = 0; i < size; i ++ )
   {
      y[ i ] = b[ i ];
      const m_int i_row_beg = M_rows_info[ i ]. first;
      const m_int i_row_end = M_rows_info[ i ]. diagonal;
      m_int col;
      for( j = i_row_beg;
           j < i_row_end && ( col = M_data[ j ]. column ) != -1;
           j ++ )
         y[ i ] -= y[ M_data[ j ]. column ] * M_data[ j ]. value;
      //for( j = 0; j < i; j ++ )
      //   y[ i ] -= y[ j ] * ( * M )( i, j );
   }
   
   dbgCout( "Solving Ux = y" );
   for( i = size - 1; i >=0 ; i -- )
   {
      x[ i ] = y[ i ];
      const m_int i_row_beg = M_rows_info[ i ]. diagonal;
      assert( i_row_beg != -1 );
      const m_int i_row_end = M_rows_info[ i ]. last;
      m_int col;
      for( j = i_row_beg + 1;
           j < i_row_end && ( col = M_data[ j ]. column ) != -1;
           j ++ )
         x[ i ] -= x[ col ] * M_data[ j ]. value;
      
      assert( M_data[ i_row_beg ]. value != 0.0 );
      x[ i ] /= M_data[ i_row_beg ]. value;
      //for( j = i + 1; j < size; j ++ )
      //   x[ i ] -= x[ j ] * ( *M )( i, j );
      //x[ i ] /= ( *M )( i, i );
   }
#ifdef ILU_DEBUG
   /*for( i = 0; i < size; i ++ )
   {
      y[ i ] = b[ i ];
      for( j = 0; j < i; j ++ )
         y[ i ] -= y[ j ] * full_M[ i * size + j ];
   }
   
   dbgCout( "Solving Ux = y" );
   for( i = size - 1; i >=0 ; i -- )
   {
      x[ i ] = y[ i ];
      for( j = i + 1; j < size; j ++ )
         x[ i ] -= x[ j ] * full_M[ i * size + j ];
      x[ i ] /= full_M[ i * size + i ];
   }*/
#endif
}

