/***************************************************************************
                          mEulerSolver.h  -  description
                             -------------------
    begin                : 2008/04/01
    copyright            : (C) 2008 by Tomá¹ Oberhuber
    email                : oberhuber@seznam.cz
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/

#ifndef mEulerSolverH
#define mEulerSolverH

#include <math.h>
#include <mcore.h>
#include "mExplicitSolver.h"

template< class GRID, class SCHEME, typename T = double > class mEulerSolver : public mExplicitSolver< GRID, SCHEME, T >
{
   public:

   mEulerSolver( const GRID& v )
   {
      k1 = new GRID( v );
      if( ! k1 )
      {
         cerr << "Unable to allocate supporting structures for the Euler solver." << endl;
         abort();
      };
      k1 -> Zeros();
   };

   mString GetType() const
   {
      T t;
      GRID grid;
      return mString( "mEulerSolver< " ) + grid. GetType() + 
             mString( ", " ) + GetParameterType( t ) + mString( " >" );
   };

   bool Solve( SCHEME& scheme,
               GRID& u,
               const double& stop_time,
               const double& max_res,
               const long int max_iter )
   {
      T* _k1 = k1 -> Data();
      T* _u = u. Data();
           
      mExplicitSolver< GRID, SCHEME, T > :: iteration = 0;
      double& _time = mExplicitSolver< GRID, SCHEME, T > :: time;  
      double& _residue = mExplicitSolver< GRID, SCHEME, T > :: residue;  
      long int& _iteration = mExplicitSolver< GRID, SCHEME, T > :: iteration;
      const double size_inv = 1.0 / ( double ) u. GetSize();
      
      T _tau = mExplicitSolver< GRID, SCHEME, T > :: tau;
      if( _time + _tau > stop_time ) _tau = stop_time - _time;
      if( _tau == 0.0 ) return true;

      if( mExplicitSolver< GRID, SCHEME, T > :: verbosity > 0 )
         mExplicitSolver< GRID, SCHEME, T > :: PrintOut();
      while( 1 )
      {
         long int i;
         const long int size = k1 -> GetSize();
         assert( size == u. GetSize() );
         
         scheme. GetExplicitRHS( _time, u, *k1 );

         double last_residue = _residue;
         double loc_residue = 0.0;
#ifdef HAVE_OPENMP
#pragma omp parallel for reduction(+:loc_residue) firstprivate( size, _u, _k1,_tau )
#endif
         for( i = 0; i < size; i ++ )
         {
            const T add = _tau * _k1[ i ];
            _u[ i ] += add; 
            loc_residue += fabs( ( double ) add );
         }
         if( _tau + _time == stop_time ) _residue = last_residue;  // fixing strange values of res. at the last iteration
         else
         {
             loc_residue /= _tau * size_inv;
             :: MPIAllreduce( loc_residue, _residue, 1, MPI_SUM, mExplicitSolver< GRID, SCHEME, T > :: solver_comm );
         }
         _time += _tau;
         _iteration ++;
         
         if( _time + _tau > stop_time )
            _tau = stop_time - _time; //we don't want to keep such tau
         else mExplicitSolver< GRID, SCHEME, T > :: tau = _tau;
         
         if( mExplicitSolver< GRID, SCHEME, T > :: verbosity > 1 )
            mExplicitSolver< GRID, SCHEME, T > :: PrintOut();
         
         if( _time == stop_time || 
             ( max_res && _residue < max_res ) )
          {
            if( mExplicitSolver< GRID, SCHEME, T > :: verbosity > 0 )
               mExplicitSolver< GRID, SCHEME, T > :: PrintOut();
             return true;
          }
         //if( max_iter && _iteration == max_iter ) return false;
      }
   };

   ~mEulerSolver()
   {
      delete k1;
   };

   protected:
   
   GRID *k1;
};

#endif
