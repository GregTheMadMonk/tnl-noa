/***************************************************************************
                          tnlGMRESSolver_impl.h  -  description
                             -------------------
    begin                : Nov 25, 2012
    copyright            : (C) 2012 by Tomas Oberhuber
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

#ifndef TNLGMRESSOLVER_IMPL_H_
#define TNLGMRESSOLVER_IMPL_H_

template< typename Matrix,
           typename Preconditioner >
tnlGMRESSolver< Matrix, Preconditioner > :: tnlGMRESSolver()
: tnlObject( "no-name" ),
  _r( "tnlGMRESSolver::_r" ),
  _w( "tnlGMRESSolver::_w" ),
  _v( "tnlGMRESSolver::_v" ),
  _M_tmp( "tnlGMRESSolver::_M_tmp" ),
  _s( "tnlGMRESSolver::_s" ),
  _cs( "tnlGMRESSolver::_cs" ),
  _sn( "tnlGMRESSolver::_sn" ),
  _H( "tnlGMRESSolver:_H" ),
  size( 0 ),
  restarting( 10 ),
  matrix( 0 ),
  preconditioner( 0 )
{
};

template< typename Matrix,
          typename Preconditioner >
tnlString
tnlGMRESSolver< Matrix, Preconditioner >::
getType() const
{
   return tnlString( "tnlGMRESSolver< " ) +
          this->matrix -> getType() + ", " +
          this->preconditioner -> getType() + " >";
}

template< typename Matrix,
          typename Preconditioner >
void
tnlGMRESSolver< Matrix, Preconditioner >::
configSetup( tnlConfigDescription& config,
             const tnlString& prefix )
{
   tnlIterativeSolver< RealType, IndexType >::configSetup( config, prefix );
   config.addEntry< int >( prefix + "gmres-restarting", "Number of iterations after which the GMRES restarts.", 10 );
}

template< typename Matrix,
          typename Preconditioner >
bool
tnlGMRESSolver< Matrix, Preconditioner >::
setup( const tnlParameterContainer& parameters,
       const tnlString& prefix )
{
   tnlIterativeSolver< RealType, IndexType >::setup( parameters, prefix );
   this->setRestarting( parameters.getParameter< int >( "gmres-restarting" ) );
}

template< typename Matrix,
           typename Preconditioner >
void tnlGMRESSolver< Matrix, Preconditioner > :: setRestarting( IndexType rest )
{
   if( size != 0 )
      setSize( size, rest );
   restarting = rest;
};

template< typename Matrix,
          typename Preconditioner >
void tnlGMRESSolver< Matrix, Preconditioner > :: setMatrix( const MatrixType& matrix )
{
   this->matrix = &matrix;
}

template< typename Matrix,
           typename Preconditioner >
void tnlGMRESSolver< Matrix, Preconditioner > :: setPreconditioner( const Preconditioner& preconditioner )
{
   this->preconditioner = &preconditioner;
}

template< typename Matrix,
          typename Preconditioner >
 template< typename Vector, typename ResidueGetter >
bool tnlGMRESSolver< Matrix, Preconditioner > :: solve( const Vector& b, Vector& x )
{
   tnlAssert( matrix, cerr << "No matrix was set in tnlGMRESSolver. Call setMatrix() before solve()." << endl );
   if( restarting <= 0 )
   {
      cerr << "I have wrong value for the restarting of the GMRES solver. It is set to " << restarting
           << ". Please set some positive value using the SetRestarting method." << endl;
      return false;
   }
   if( ! setSize( matrix -> getRows(), restarting ) )
   {
       cerr << "I am not able to allocate enough memory for the GMRES solver. You may try to decrease the restarting parameter." << endl;
       return false;
   }

   IndexType _size = size;

   RealType *r = _r. getData();
   RealType *w = _w. getData();
   RealType *s = _s. getData();
   RealType *cs = _cs. getData();
   RealType *sn = _sn. getData();
   RealType *v = _v. getData();
   RealType *H = _H. getData();
   RealType *M_tmp = _M_tmp. getData();

   RealType normb( 0.0 ), beta( 0.0 );
   //T normb( 0.0 ), beta( 0.0 ); does not work with openmp yet
   /****
    * 1. Solve r from M r = b - A x_0
    */
   if( preconditioner )
   {
      this->preconditioner->solve( b, _M_tmp );
      normb = _M_tmp. lpNorm( ( RealType ) 2.0 );

      matrix -> vectorProduct( x, _M_tmp );
      _M_tmp.addVector( b, ( RealType ) 1.0, -1.0 );
      /*for( i = 0; i < size; i ++ )
         M_tmp[ i ] = b[ i ] - M_tmp[ i ];*/

      this->preconditioner->solve( _M_tmp, _r );
      beta = _r. lpNorm( ( RealType ) 2.0 );
   }
   else
   {
      matrix -> vectorProduct( x, _r );      
      normb = b. lpNorm( ( RealType ) 2.0 );
      _r. addVector( b, ( RealType ) 1.0, -1.0 );
      beta = _r. lpNorm( ( RealType ) 2.0 );
      //cout << "x = " << x << endl;
   }
   
    //cout << "norm b = " << normb << endl;
    //cout << " beta = " << beta << endl;


   if( normb == 0.0 ) normb = 1.0;

   this->resetIterations();
   this->setResidue( beta / normb );

   tnlSharedVector< RealType, DeviceType, IndexType > vi;
   //vi. setName( "tnlGMRESSolver::vi" );
   tnlSharedVector< RealType, DeviceType, IndexType > vk;
   //vk. setName( "tnlGMRESSolver::vk" );
   while( this->nextIteration() )
   {
      const IndexType m = restarting;
      for( IndexType i = 0; i < m + 1; i ++ )
         H[ i ] = s[ i ] = cs[ i ] = sn[ i ] = 0.0;

      /****
       * v = 0
       */
      _v. setValue( ( RealType ) 0.0 );

      /***
       * v_0 = r / | r | =  1.0 / beta * r
       */
      vi. bind( _v. getData(), size );
      vi. addVector( _r, ( RealType ) 1.0 / beta );

      _s. setValue( ( RealType ) 0.0 );
      _s[ 0 ] = beta;



      /****
       * Starting m-loop
       */
      for( IndexType i = 0; i < m && this->getIterations() <= this->getMaxIterations(); i++ )
      {
         vi. bind( &( _v. getData()[ i * size ] ), size );
         /****
          * Solve w from M w = A v_i
          */
         if( preconditioner )
         {
            matrix->vectorProduct( vi, _M_tmp );
            this->preconditioner->solve( _M_tmp, _w );
         }
         else
             matrix -> vectorProduct( vi, _w );
         
         //cout << " i = " << i << " vi = " << vi << endl;

         for( IndexType k = 0; k <= i; k++ )
            H[ k + i * ( m + 1 ) ] = 0.0;
         for( IndexType l = 0; l < 2; l++ )
            for( IndexType k = 0; k <= i; k++ )
            {
               vk. bind( &( _v. getData()[ k * _size ] ), _size );
               /***
                * H_{k,i} = ( w, v_k )
                */
               RealType H_k_i = _w. scalarProduct( vk );
               H[ k + i * ( m + 1 ) ] += H_k_i;           

               /****
                * w = w - H_{k,i} v_k
                */
               _w. addVector( vk, -H_k_i );

               //cout << "H_ki = " << H_k_i << endl;
               //cout << "w = " << _w << endl;
            }
         /***
          * H_{i+1,i} = |w|
          */
         RealType normw = _w. lpNorm( ( RealType ) 2.0 );
         H[ i + 1 + i * ( m + 1 ) ] = normw;

         //cout << "normw = " << normw << endl;
         
         /***
          * v_{i+1} = w / |w|
          */
         vi. bind( &( _v. getData()[ ( i + 1 ) * size ] ), size );
         vi. addVector( _w, ( RealType ) 1.0 / normw );
         
         //cout << "vi = " << vi << endl;

         /****
          * Applying the Givens rotations
          */
         for( IndexType k = 0; k < i; k++ )
            applyPlaneRotation( H[ k + i * ( m + 1 )],
                                H[ k + 1 + i * ( m + 1 ) ],
                                cs[ k ],
                                sn[ k ] );

         generatePlaneRotation( H[ i + i * ( m + 1 ) ],
                                H[ i + 1 + i * ( m + 1 ) ],
                                cs[ i ],
                                sn[ i ]);
         applyPlaneRotation( H[ i + i * ( m + 1 ) ],
                             H[ i + 1 + i * ( m + 1 ) ],
                             cs[ i ],
                             sn[ i ]);
         applyPlaneRotation( s[ i ],
                             s[ i + 1 ],
                             cs[ i ],
                             sn[ i ] );

         this->setResidue( fabs( s[ i + 1 ] ) / normb );
         this->refreshSolverMonitor();

         if( this->getResidue() < this->getConvergenceResidue() )
         {
            update( i, m, _H, _s, _v, x );
            return true;
         }
         if( ! this->nextIteration() )
            return false;
      }
      update( m - 1, m, _H, _s, _v, x );

      /****
       * r = M.solve(b - A * x);
       */
      beta = 0.0;
      if( preconditioner )
      {
         matrix -> vectorProduct( x, _M_tmp );
         for( IndexType i = 0; i < _size; i ++ )
            M_tmp[ i ] = b[ i ] - M_tmp[ i ];
         //preconditioner -> solve( M_tmp, r );
         for( IndexType i = 0; i < _size; i ++ )
            beta += r[ i ] * r[ i ];
      }
      else
      {
         matrix -> vectorProduct( x, _r );
         _r.addVector( b, ( RealType ) 1.0, -1.0 );
         beta = _r. lpNorm( ( RealType ) 2.0 );
      }
      this->setResidue( beta / normb );

      //cout << " x = " << x << endl;
      //cout << " beta = " << beta << endl;
      //cout << "residue = " << beta / normb << endl;

   }
   this->refreshSolverMonitor();
   return this->checkConvergence();
};

template< typename Matrix,
          typename Preconditioner >
tnlGMRESSolver< Matrix, Preconditioner > :: ~tnlGMRESSolver()
{
};

template< typename Matrix,
          typename Preconditioner >
   template< typename Vector >
void tnlGMRESSolver< Matrix, Preconditioner > :: update( IndexType k,
                                                         IndexType m,
                                                         const tnlVector< RealType, tnlHost, IndexType >& H,
                                                         const tnlVector< RealType, tnlHost, IndexType >& s,
                                                         tnlVector< RealType, DeviceType, IndexType >& v,
                                                         Vector& x )
{
   //dbgFunctionName( "tnlGMRESSolver", "Update" );
   tnlVector< RealType, tnlHost, IndexType > y( "tnlGMRESSolver::update:y" );
   y. setSize( m + 1 );

   IndexType i, j;
   for( i = 0; i <= m ; i ++ )
      y[ i ] = s[ i ];

   //dbgCout_ARRAY( y, m + 1 );
   // Backsolve:
   for( i = k; i >= 0; i--)
   {
      y[ i ] /= H[ i + i * ( m + 1 ) ];
      for( j = i - 1; j >= 0; j--)
         y[ j ] -= H[ j + i * ( m + 1 ) ] * y[ i ];
   }
   //dbgCout_ARRAY( y, m + 1 );

   tnlSharedVector< RealType, DeviceType, IndexType > vi;
   vi. setName( "tnlGMRESSolver::update:vi" );
   for( i = 0; i <= k; i++)
   {
      vi. bind( &( v. getData()[ i * this->size ] ), x. getSize() );
      x. addVector( vi, y[ i ] );
   }
};

template< typename Matrix,
          typename Preconditioner >
void tnlGMRESSolver< Matrix, Preconditioner > :: generatePlaneRotation( RealType &dx,
                                                                        RealType &dy,
                                                                        RealType &cs,
                                                                        RealType &sn )
{
   if( dy == 0.0 )
   {
      cs = 1.0;
      sn = 0.0;
   }
   else
      if( fabs( dy ) > fabs( dx ) )
      {
         RealType temp = dx / dy;
         sn = 1.0 / sqrt( 1.0 + temp * temp );
         cs = temp * sn;
      }
      else
      {
         RealType temp = dy / dx;
         cs = 1.0 / sqrt( 1.0 + temp * temp );
         sn = temp * cs;
      }
};

template< typename Matrix,
          typename Preconditioner >
void tnlGMRESSolver< Matrix, Preconditioner > :: applyPlaneRotation( RealType &dx,
                                                                     RealType &dy,
                                                                     RealType &cs,
                                                                     RealType &sn )
{
   RealType temp  =  cs * dx + sn * dy;
   dy =  cs * dy - sn * dx;
   dx = temp;
};

template< typename Matrix,
          typename Preconditioner >
bool tnlGMRESSolver< Matrix, Preconditioner > :: setSize( IndexType _size, IndexType m )
{
   if( size == _size && restarting == m ) return true;
   size = _size;
   restarting = m;
   if( ! _r. setSize( size ) ||
       ! _w. setSize( size ) ||
       ! _s. setSize( restarting + 1 ) ||
       ! _cs. setSize( restarting + 1 ) ||
       ! _sn. setSize( restarting + 1 ) ||
       ! _v. setSize( size * ( restarting + 1 ) ) ||
       ! _H. setSize( ( restarting + 1 ) * restarting ) ||
       ! _M_tmp. setSize( size ) )
   {
      cerr << "I could not allocated all supporting arrays for the CG solver." << endl;
      return false;
   }
   return true;
};

#endif /* TNLGMRESSOLVER_IMPL_H_ */
