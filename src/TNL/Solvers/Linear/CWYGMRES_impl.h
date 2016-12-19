#pragma once

#include <type_traits>

#include <TNL/Containers/Algorithms/Multireduction.h>
#include <TNL/Matrices/MatrixOperations.h>

#include "CWYGMRES.h"

namespace TNL {
namespace Solvers {
namespace Linear {

template< typename Matrix,
          typename Preconditioner >
CWYGMRES< Matrix, Preconditioner >::
CWYGMRES()
: size( 0 ),
  ldSize( 0 ),
  restarting( 10 )
{
}

template< typename Matrix,
          typename Preconditioner >
CWYGMRES< Matrix, Preconditioner >::
~CWYGMRES()
{
}

template< typename Matrix,
          typename Preconditioner >
String
CWYGMRES< Matrix, Preconditioner >::
getType() const
{
   return String( "CWYGMRES< " ) +
          this->matrix -> getType() + ", " +
          this->preconditioner -> getType() + " >";
}

template< typename Matrix,
          typename Preconditioner >
void
CWYGMRES< Matrix, Preconditioner >::
configSetup( Config::ConfigDescription& config,
             const String& prefix )
{
   //IterativeSolver< RealType, IndexType >::configSetup( config, prefix );
   config.addEntry< int >( prefix + "gmres-restarting", "Number of iterations after which the CWYGMRES restarts.", 10 );
}

template< typename Matrix,
          typename Preconditioner >
bool
CWYGMRES< Matrix, Preconditioner >::
setup( const Config::ParameterContainer& parameters,
       const String& prefix )
{
   IterativeSolver< RealType, IndexType >::setup( parameters, prefix );
   this->setRestarting( parameters.getParameter< int >( "gmres-restarting" ) );
   return true;
}

template< typename Matrix,
          typename Preconditioner >
void
CWYGMRES< Matrix, Preconditioner >::
setRestarting( IndexType rest )
{
   if( size != 0 )
      setSize( size, rest );
   restarting = rest;
}

template< typename Matrix,
          typename Preconditioner >
void
CWYGMRES< Matrix, Preconditioner >::
setMatrix( const MatrixPointer& matrix )
{
   this->matrix = matrix;
}

template< typename Matrix,
          typename Preconditioner >
void
CWYGMRES< Matrix, Preconditioner >::
setPreconditioner( const PreconditionerPointer& preconditioner )
{
   this->preconditioner = preconditioner;
}

template< typename Matrix,
          typename Preconditioner >
   template< typename Vector, typename ResidueGetter >
bool
CWYGMRES< Matrix, Preconditioner >::
solve( const Vector& b, Vector& x )
{
   TNL_ASSERT( matrix, std::cerr << "No matrix was set in CWYGMRES. Call setMatrix() before solve()." << std::endl );
   if( restarting <= 0 )
   {
      std::cerr << "I have wrong value for the restarting of the CWYGMRES solver. It is set to " << restarting
           << ". Please set some positive value using the SetRestarting method." << std::endl;
      return false;
   }
   if( ! setSize( matrix -> getRows(), restarting ) )
   {
      std::cerr << "I am not able to allocate enough memory for the CWYGMRES solver. You may try to decrease the restarting parameter." << std::endl;
       return false;
   }

   RealType normb( 0.0 ), beta( 0.0 );
   /****
    * 1. Solve r from M r = b - A x_0
    */
   if( preconditioner )
   {
      this->preconditioner->solve( b, _M_tmp );
      normb = _M_tmp.lpNorm( ( RealType ) 2.0 );

      matrix->vectorProduct( x, _M_tmp );
      _M_tmp.addVector( b, ( RealType ) 1.0, -1.0 );

      this->preconditioner->solve( _M_tmp, r );
   }
   else
   {
      matrix->vectorProduct( x, r );
      normb = b.lpNorm( ( RealType ) 2.0 );
      r.addVector( b, ( RealType ) 1.0, -1.0 );
   }
   beta = r.lpNorm( ( RealType ) 2.0 );

   //cout << "norm b = " << normb << endl;
   //cout << " beta = " << beta << endl;


   if( normb == 0.0 )
      normb = 1.0;

   this->resetIterations();
   this->setResidue( beta / normb );

   DeviceVector vi, vk;
   while( this->checkNextIteration() )
   {
      const IndexType m = restarting;

      /***
       * z = r / | r | =  1.0 / beta * r
       */
//      z.addVector( r, ( RealType ) 1.0 / beta, ( RealType ) 0.0 );
      z = r;

      H.setValue( 0.0 );
      s.setValue( 0.0 );
      T.setValue( 0.0 );

      // NOTE: this is unstable, s[0] is set later
//      s[ 0 ] = beta;

      /****
       * Starting m-loop
       */
      for( IndexType i = 0; i <= m && this->nextIteration(); i++ )
      {
//         cout << "==== i = " << i << " ====" << endl;

         /****
          * Generate new Hauseholder transformation from vector z.
          */
         hauseholder_generate( Y, T, i, z );

         if( i == 0 ) {
            /****
             * s = e_1^T * P_i * z
             */
            hauseholder_apply_trunc( s, Y, T, i, z );
         }
         else {
            /***
             * H_{i-1} = P_i * z
             */
            HostVector h;
            h.bind( &H.getData()[ (i - 1) * (m + 1) ], m + 1 );
            hauseholder_apply_trunc( h, Y, T, i, z );
         }

         /***
          * Generate new basis vector v_i, using the compact WY representation:
          *     v_i = (I - Y_i * T_i Y_i^T) * e_i
          */
         vi.bind( &V.getData()[ i * ldSize ], size );
         hauseholder_cwy( vi, Y, T, i );

         if( i < m ) {
            /****
             * Solve w from M w = A v_i
             */
            if( preconditioner )
            {
               matrix->vectorProduct( vi, _M_tmp );
               this->preconditioner->solve( _M_tmp, w );
            }
            else
                matrix -> vectorProduct( vi, w );

            /****
             * Apply all previous Hauseholder transformations, using the compact WY representation:
             *     z = (I - Y_i * T_i^T * Y_i^T) * w
             */
            hauseholder_cwy_transposed( z, Y, T, i, w );
         }

//         cout << "vi.norm = " << vi.lpNorm( 2.0 ) << endl;
//         cout << "H (before rotations) = " << endl;
//         for( int i = 0; i <= m; i++ ) {
//             for( int j = 0; j < m; j++ )
//                 cout << H[ i + j * (m+1) ] << "\t";
//             cout << endl;
//         }
//         cout << "s = " << s << endl;

         /****
          * Applying the Givens rotations
          */
         if( i > 0 ) {
            for( IndexType k = 0; k < i - 1; k++ )
               applyPlaneRotation( H[ k     + (i - 1) * (m + 1) ],
                                   H[ k + 1 + (i - 1) * (m + 1) ],
                                   cs[ k ],
                                   sn[ k ] );

            if( H[ i + (i - 1) * (m + 1) ] != 0.0 ) {
               generatePlaneRotation( H[ i - 1 + (i - 1) * (m + 1) ],
                                      H[ i     + (i - 1) * (m + 1) ],
                                      cs[ i - 1 ],
                                      sn[ i - 1 ]);
               applyPlaneRotation( H[ i - 1 + (i - 1) * (m + 1) ],
                                   H[ i     + (i - 1) * (m + 1) ],
                                   cs[ i - 1 ],
                                   sn[ i - 1 ]);
               applyPlaneRotation( s[ i - 1 ],
                                   s[ i     ],
                                   cs[ i - 1 ],
                                   sn[ i - 1 ] );
            }
         }

//         cout << "H (after rotations) = " << endl;
//         for( int i = 0; i <= m; i++ ) {
//             for( int j = 0; j < m; j++ )
//                 cout << H[ i + j * (m+1) ] << "\t";
//             cout << endl;
//         }
//         cout << "s (after rotations) = " << s << endl;
//         cout << "residue / normb = " << std::fabs( s[ i ] ) / normb << endl;

//         for( int k = 0; k <= i; k++ ) {
//            vk.bind( &V.getData()[ k * ldSize ], size );
//            cout << "(v" << k << ",v" << i << ") = " << vk.scalarProduct( vi ) << endl;
//         }

         this->setResidue( std::fabs( s[ i ] ) / normb );
         if( ! this->checkNextIteration() ) {
            update( i - 1, m, H, s, V, x );
            this->refreshSolverMonitor( true );
            return this->checkConvergence();
         }
         else {
            this->refreshSolverMonitor();
         }
      }
      update( m - 1, m, H, s, V, x );

      /****
       * r = M.solve(b - A * x);
       */
      if( preconditioner )
      {
         matrix->vectorProduct( x, _M_tmp );
         _M_tmp.addVector( b, ( RealType ) 1.0, -1.0 );
         preconditioner->solve( _M_tmp, r );
      }
      else
      {
         matrix->vectorProduct( x, r );
         r.addVector( b, ( RealType ) 1.0, -1.0 );
      }
      beta = r.lpNorm( ( RealType ) 2.0 );
      this->setResidue( beta / normb );

//      cout << " x = " << x << endl;
//      cout << " beta = " << beta << endl;
//      cout << "residue = " << beta / normb << endl;

   }
   this->refreshSolverMonitor( true );
   return this->checkConvergence();
}

#ifdef HAVE_CUDA
template< typename DestinationElement,
          typename SourceElement,
          typename Index >
__global__ void
copyTruncatedVectorKernel( DestinationElement* destination,
                           const SourceElement* source,
                           const Index from,
                           const Index size )
{
   Index elementIdx = blockIdx.x * blockDim.x + threadIdx.x;
   const Index gridSize = blockDim.x * gridDim.x;

   while( elementIdx < from )
   {
      destination[ elementIdx ] = (DestinationElement) 0.0;
      elementIdx += gridSize;
   }
   while( elementIdx < size )
   {
      destination[ elementIdx ] = source[ elementIdx ];
      elementIdx += gridSize;
   }
}
#endif

template< typename Matrix,
          typename Preconditioner >
void
CWYGMRES< Matrix, Preconditioner >::
hauseholder_generate( DeviceVector& Y,
                      HostVector& T,
                      const int& i,
                      DeviceVector& z )
{
   DeviceVector y_i;
   y_i.bind( &Y.getData()[ i * ldSize ], size );

   // XXX: the upper-right triangle of Y will be full of zeros, which can be exploited for optimization
   if( std::is_same< DeviceType, Devices::Host >::value ) {
      for( IndexType j = 0; j < size; j++ ) {
         if( j < i )
            y_i[ j ] = 0.0;
         else
            y_i[ j ] = z[ j ];
      }
   }
   if( std::is_same< DeviceType, Devices::Cuda >::value ) {
#ifdef HAVE_CUDA
      dim3 blockSize( 256 );
      dim3 gridSize;
      gridSize.x = min( Devices::Cuda::getMaxGridSize(), Devices::Cuda::getNumberOfBlocks( size, blockSize.x ) );

      copyTruncatedVectorKernel<<< gridSize, blockSize >>>( y_i.getData(),
                                                            z.getData(),
                                                            i,
                                                            size );
      checkCudaDevice;
#else
      CudaSupportMissingMessage;
#endif
   }

   // norm of the TRUNCATED vector z
   const RealType normz = y_i.lpNorm( 2.0 );
   const RealType y_ii = y_i.getElement( i );
   if( y_ii > 0.0 ) {
      y_i.setElement( i, y_ii + normz );
   }
   else {
      y_i.setElement( i, y_ii - normz );
   }

   // compute the norm of the y_i vector; equivalent to this calculation by definition:
   //       const RealType norm_yi = y_i.lpNorm( 2.0 );
   const RealType norm_yi = std::sqrt( 2 * (normz * normz + std::fabs( y_ii ) * normz) );

   // XXX: normalization is slower, but more stable
//   y_i *= 1.0 / norm_yi;
//   const RealType t_i = 2.0;
   // assuming it's stable enough...
   const RealType t_i = 2.0 / (norm_yi * norm_yi);

   T[ i + i * (restarting + 1) ] = t_i;
   if( i > 0 ) {
      // aux = Y_{i-1}^T * y_i
      RealType aux[ i ];
      Containers::Algorithms::tnlParallelReductionScalarProduct< RealType, IndexType > scalarProduct;
      if( ! Containers::Algorithms::Multireduction< DeviceType >::reduce
               ( scalarProduct,
                 i,
                 size,
                 Y.getData(),
                 ldSize,
                 y_i.getData(),
                 aux ) )
      {
         std::cerr << "multireduction failed" << std::endl;
         throw 1;
      }

      // [T_i]_{0..i-1} = - T_{i-1} * t_i * aux
      for( int k = 0; k < i; k++ ) {
         T[ k + i * (restarting + 1) ] = 0.0;
         for( int j = k; j < i; j++ )
            T[ k + i * (restarting + 1) ] -= T[ k + j * (restarting + 1) ] * (t_i * aux[ j ]);
      }
   }
}

template< typename Matrix,
          typename Preconditioner >
void
CWYGMRES< Matrix, Preconditioner >::
hauseholder_apply_trunc( HostVector& out,
                         DeviceVector& Y,
                         HostVector& T,
                         const int& i,
                         DeviceVector& z )
{
   DeviceVector y_i;
   y_i.bind( &Y.getData()[ i * ldSize ], size );

   const RealType aux = T[ i + i * (restarting + 1) ] * y_i.scalarProduct( z );
   if( std::is_same< DeviceType, Devices::Host >::value ) {
      for( int k = 0; k <= i; k++ )
         out[ k ] = z[ k ] - y_i[ k ] * aux;
   }
   if( std::is_same< DeviceType, Devices::Cuda >::value ) {
      // copy part of y_i to buffer on host
      // here we duplicate the upper (m+1)x(m+1) submatrix of Y on host for fast access
      RealType* host_yi = &YL[ i * (restarting + 1) ];
      RealType host_z[ i + 1 ];
      if( ! Containers::Algorithms::ArrayOperations< Devices::Host, Devices::Cuda >::copyMemory< RealType, RealType, IndexType >( host_yi, y_i.getData(), restarting + 1 ) ||
          ! Containers::Algorithms::ArrayOperations< Devices::Host, Devices::Cuda >::copyMemory< RealType, RealType, IndexType >( host_z, z.getData(), i + 1 ) )
      {
         std::cerr << "Failed to copy part of device vectors y_i or z to host buffer." << std::endl;
         throw 1;
      }
      for( int k = 0; k <= i; k++ )
         out[ k ] = host_z[ k ] - host_yi[ k ] * aux;
   }
}

template< typename Matrix,
          typename Preconditioner >
void
CWYGMRES< Matrix, Preconditioner >::
hauseholder_cwy( DeviceVector& v,
                 DeviceVector& Y,
                 HostVector& T,
                 const int& i )
{
   // aux = Y_i^T * e_i
   RealType aux[ i + 1 ];
   if( std::is_same< DeviceType, Devices::Host >::value ) {
      for( int k = 0; k <= i; k++ )
         aux[ k ] = Y[ i + k * ldSize ];
   }
   if( std::is_same< DeviceType, Devices::Cuda >::value ) {
      // the upper (m+1)x(m+1) submatrix of Y is duplicated on host for fast access
      for( int k = 0; k <= i; k++ )
         aux[ k ] = YL[ i + k * (restarting + 1) ];
   }

   // aux = T_i * aux
   // Note that T_i is upper triangular, so we can overwrite the aux vector with the result in place
   for( int k = 0; k <= i; k++ ) {
      RealType aux2 = 0.0;
      for( int j = k; j <= i; j++ )
         aux2 += T[ k + j * (restarting + 1) ] * aux[ j ];
      aux[ k ] = aux2;
   }

   // v = e_i - Y_i * aux
   MatrixOperations< DeviceType >::gemv( size, i + 1,
                                         -1.0, Y.getData(), ldSize, aux,
                                         0.0, v.getData() );
   v.setElement( i, 1.0 + v.getElement( i ) );
}

template< typename Matrix,
          typename Preconditioner >
void
CWYGMRES< Matrix, Preconditioner >::
hauseholder_cwy_transposed( DeviceVector& z,
                            DeviceVector& Y,
                            HostVector& T,
                            const int& i,
                            DeviceVector& w )
{
   // aux = Y_i^T * w
   RealType aux[ i + 1 ];
   Containers::Algorithms::tnlParallelReductionScalarProduct< RealType, IndexType > scalarProduct;
   if( ! Containers::Algorithms::Multireduction< DeviceType >::reduce
            ( scalarProduct,
              i + 1,
              size,
              Y.getData(),
              ldSize,
              w.getData(),
              aux ) )
   {
      std::cerr << "multireduction failed" << std::endl;
      throw 1;
   }

   // aux = T_i^T * aux
   // Note that T_i^T is lower triangular, so we can overwrite the aux vector with the result in place
   for( int k = i; k >= 0; k-- ) {
      RealType aux2 = 0.0;
      for( int j = 0; j <= k; j++ )
         aux2 += T[ j + k * (restarting + 1) ] * aux[ j ];
      aux[ k ] = aux2;
   }

   // z = w - Y_i * aux
   z = w;
   MatrixOperations< DeviceType >::gemv( size, i + 1,
                                         -1.0, Y.getData(), ldSize, aux,
                                         1.0, z.getData() );
}

template< typename Matrix,
          typename Preconditioner >
   template< typename Vector >
void
CWYGMRES< Matrix, Preconditioner >::
update( IndexType k,
        IndexType m,
        const HostVector& H,
        const HostVector& s,
        DeviceVector& v,
        Vector& x )
{
//   Containers::Vector< RealType, Devices::Host, IndexType > y;
//   y.setSize( m + 1 );
   RealType y[ m + 1 ];

   IndexType i, j;
   for( i = 0; i <= m ; i ++ )
      y[ i ] = s[ i ];

   // Backsolve:
   for( i = k; i >= 0; i--) {
      if( H[ i + i * ( m + 1 ) ] == 0 ) {
//         for( int _i = 0; _i <= i; _i++ ) {
//             for( int _j = 0; _j < i; _j++ )
//                 cout << H[ _i + _j * (m+1) ] << "  ";
//             cout << endl;
//         }
         std::cerr << "H.norm = " << H.lpNorm( 2.0 ) << std::endl;
         std::cerr << "s = " << s << std::endl;
         std::cerr << "k = " << k << ", m = " << m << std::endl;
         throw 1;
      }
      y[ i ] /= H[ i + i * ( m + 1 ) ];
      for( j = i - 1; j >= 0; j--)
         y[ j ] -= H[ j + i * ( m + 1 ) ] * y[ i ];
   }

   // x = V * y + x
   MatrixOperations< DeviceType >::gemv( size, k + 1,
                                         1.0, v.getData(), ldSize, y,
                                         1.0, x.getData() );
}

template< typename Matrix,
          typename Preconditioner >
void
CWYGMRES< Matrix, Preconditioner >::
generatePlaneRotation( RealType& dx,
                       RealType& dy,
                       RealType& cs,
                       RealType& sn )
{
   if( dy == 0.0 )
   {
      cs = 1.0;
      sn = 0.0;
   }
   else
      if( std::fabs( dy ) > std::fabs( dx ) )
      {
         RealType temp = dx / dy;
         sn = 1.0 / std::sqrt( 1.0 + temp * temp );
         cs = temp * sn;
      }
      else
      {
         RealType temp = dy / dx;
         cs = 1.0 / std::sqrt( 1.0 + temp * temp );
         sn = temp * cs;
      }
}

template< typename Matrix,
          typename Preconditioner >
void CWYGMRES< Matrix, Preconditioner > ::
applyPlaneRotation( RealType& dx,
                    RealType& dy,
                    RealType& cs,
                    RealType& sn )
{
   RealType temp  =  cs * dx + sn * dy;
   dy =  cs * dy - sn * dx;
   dx = temp;
}

template< typename Matrix,
          typename Preconditioner >
bool CWYGMRES< Matrix, Preconditioner > :: setSize( IndexType _size, IndexType m )
{
   if( size == _size && restarting == m ) return true;
   size = _size;
   if( std::is_same< DeviceType, Devices::Cuda >::value )
      // align each column to 256 bytes - optimal for CUDA
      ldSize = roundToMultiple( size, 256 / sizeof( RealType ) );
   else
       // on the host, we add 1 to disrupt the cache false-sharing pattern
      ldSize = roundToMultiple( size, 256 / sizeof( RealType ) ) + 1;
   restarting = m;
   if( ! r.setSize( size ) ||
       ! z.setSize( size ) ||
       ! w.setSize( size ) ||
       ! V.setSize( ldSize * ( restarting + 1 ) ) ||
       ! Y.setSize( ldSize * ( restarting + 1 ) ) ||
       ! T.setSize( (restarting + 1) * (restarting + 1) ) ||
       ! YL.setSize( (restarting + 1) * (restarting + 1) ) ||
       ! cs.setSize( restarting + 1 ) ||
       ! sn.setSize( restarting + 1 ) ||
       ! H.setSize( ( restarting + 1 ) * restarting ) ||
       ! s.setSize( restarting + 1 ) ||
       ! _M_tmp.setSize( size ) )
   {
      std::cerr << "I could not allocate all supporting arrays for the CWYGMRES solver." << std::endl;
      return false;
   }
   return true;
}

} // namespace Linear
} // namespace Solvers
} // namespace TNL
