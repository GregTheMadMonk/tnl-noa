/***************************************************************************
                          tnlILUPreconditioner.h  -  description
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

#ifndef tnlILUPreconditionerH
#define tnlILUPreconditionerH

#include <matrix/tnlPreconditioner.h>
#include <matrix/tnlCSRMatrix.h>
#include <debug/tnlDebug.h>

//#define ILU_DEBUG

template< typename T > class tnlILUPreconditioner : public tnlPreconditioner< T >
{
   public:
   
   tnlILUPreconditioner( const int _size,
                       const int initial_size,
                       const int segment_size,
                       const int init_row_elements = 0 )
   : size( _size )
   {
#ifdef ILU_DEBUG
      full_M = new T[ size * size ];
      ilu_check = new T[ size * size ];
#endif
      y = new T[ size ];
      M = new tnlCSRMatrix< T >( size, initial_size, segment_size, init_row_elements );
   }

   tnlString GetType() const
   {
      T t;
      return tnlString( "tnlILUPreconditioner< " ) + tnlString( GetParameterType( t ) ) + tnlString( " >" );
   };


   const tnlCSRMatrix< T >* Data() const
   {
      return M;
   }

   bool Init( const tnlCSRMatrix< T >& A, const T& threshold )
   {
      dbgFunctionName( "tnlILUPreconditioner", "Init" );
      assert( A. GetSize() == M -> GetSize() );
      int non_zero_elements( 0 );
#ifdef CSR_MATRIX_TUNING
      M -> ResetStatistics();
#endif
      int i, j, k;
      const tnlCSRMatrixElement< T > *A_data;
      const tnlCSRMatrixRowInfo *A_rows_info;
      A. Data( A_data, A_rows_info );
      
#ifdef ILU_DEBUG
      // 1. Copy data of A to full_M
      bzero( full_M, size * size * sizeof( T ) );
      for( i = 0; i < size; i ++ )
         for( j = A_rows_info[ i ]. first; j < A_rows_info[ i ]. last; j ++ )
         {
            int col = A_data[ j ]. column;
            if( col != -1 )
               full_M[ i * size + col ] = A_data[ j ]. value;
         }
#endif

      M -> Reset();
      tnlCSRMatrixElement< T >* M_data;
      const tnlCSRMatrixRowInfo *M_rows_info;
      // 2. Processing IKJ version of ILU factorisation
      // For i = 0, ... , N
      for( i = 0; i < size; i ++ )
      {
         // y = a_i*  - i-th row
         bzero( y, size * sizeof( T ) );
         T y_norm( 0.0 );
         const long a_row_beg = A_rows_info[ i ]. first;
         const long a_row_end = A_rows_info[ i ]. last;
         for( k = a_row_beg; k < a_row_end; k ++ )
         {
            const int col = A_data[ k ]. column;
            const T& a_ik = A_data[ k ]. value;
            if( col == -1 ) break;
            y[ col ] = a_ik;
            y_norm += a_ik * a_ik;
         }
         // get 2-norm of the i-th row for the dropping rule on y_k
         y_norm = sqrt( y_norm );
         const T threshold_i = y_norm * threshold;
         M -> Data( M_data, M_rows_info );
         for( k = 0; k < i; k ++ )
         {
            const T& m_kk = M -> GetElement( k, k );
            if( m_kk != 0.0 )
               y[ k ] /= m_kk;
            if( fabs( y[ k ] ) < threshold_i ) y[ k ] = 0.0;
            if( y[ k ] != 0.0 )
            {
               const int row_beg = M_rows_info[ k ]. diagonal + 1;
               const int row_end = M_rows_info[ k ]. last;
               int col;
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
                M -> SetElement( i, j , y[ j ] );
                non_zero_elements ++;
            }
            if( i == j && M -> GetElement( i, i ) == 0.0 )
            {
               cerr << endl << "Zero pivot appears at line " << i << "!" << endl;
               M -> SetElement( i, i, 1.0 );
               abort();
               //return 0;
            }
         }
        cout << "Computing ILUT ... " << 100.0 * ( float ) ( i + 1 ) / ( float ) ( size ) << "%    \r" << flush; 
         dbgExpr( i );
         //dbgExpr( * M );
      }
      cout << non_zero_elements / size << " elems. per line in average ";
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
   cout << "LU: " << ( float ) i / ( float ) size * 100.0 << " % done   \r" << flush;
         // y = a_i*  - i-th row
         bzero( y, size * sizeof( T ) );
         T y_norm( 0.0 );
         for( k = 0; k < size; k ++ )
         {
            const T& m_ik = full_M[ i * size + k ];
            y[ k ] = m_ik;
            y_norm += m_ik * m_ik;
            //cout << i << ", " << k << " <- " << y[ k ] << endl;
         }
         // get 2-norm of the i-th row for the dropping rule on y_k
         y_norm = sqrt( y_norm );
         const T threshold_i = y_norm * threshold;
         for( k = 0; k < i; k ++ )
         {
            const T& m_kk = full_M[ k * size + k ];
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
            int l = Min( i, j );
            T sum( 0.0 );
            for( k = 0; k <= l; k ++ )
               if( k < i )
                  sum += full_M[ i * size + k ] * full_M[ k * size + j ];
               else
                  sum += full_M[ i * size + j ];  // L has 1 on diagonal
            if( fabs( sum ) < 1.0e-15 ) sum = 0.0;
            ilu_check[ i * size + j ] = sum;
         }
      cout << endl;
      T ilu_diff( 0.0 );
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
         int i_th_row_beg = M_row_ind[ i ];
         int i_th_row_end = M_row_ind[ i + 1 ];
         int k_ind;
         for( k_ind = i_th_row_beg; k_ind < i_th_row_end; k_ind ++ )
         {
            k = M_data[ k_ind ]. column;
            if( k == -1 ) break;
            // a_ik := a_ik / a_kk
            M_data[ k_ind ]. value /= M_data[ M_diag_ind[ k ] ]. value;
            const T& a_ik = M_data[ k_ind ]. value;

            // For j = k + 1 ... n - 1 and for ( i, j ) non-zero
            // = take non-zero entries at the i-th row behind the diagonal entry
            int j_ind;
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

   
   bool Solve( const T* b, T* x ) const
   {
      dbgFunctionName( "tnlILUPreconditioner", "Solve" );
      const int size = M -> GetSize();
      const tnlCSRMatrixElement< T >* M_data;
      const tnlCSRMatrixRowInfo *M_rows_info;
      M -> Data( M_data, M_rows_info );
      int i, j;
      
      dbgCout( "Solving Ly = b" );
      // L is unit lower triangular
      for( i = 0; i < size; i ++ )
      {
         y[ i ] = b[ i ];
         const int i_row_beg = M_rows_info[ i ]. first;
         const int i_row_end = M_rows_info[ i ]. diagonal;
         int col;
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
         const int i_row_beg = M_rows_info[ i ]. diagonal;
         assert( i_row_beg != -1 );
         const int i_row_end = M_rows_info[ i ]. last;
         int col;
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
      return true;
   }
   
   ~tnlILUPreconditioner()
   {
      if( M ) delete M;
      if( y ) delete[] y;
#ifdef ILU_DEBUG
      if( full_M ) delete[] full_M;
      if( ilu_check ) delete[] ilu_check;
#endif
   }

   protected:

   tnlCSRMatrix< T >* M;

   T* y;

   int size;
   
#ifdef ILU_DEBUG
   T* full_M, *ilu_check;
#endif
};

#endif
