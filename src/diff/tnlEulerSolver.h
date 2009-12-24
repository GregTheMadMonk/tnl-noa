/***************************************************************************
                          tnlEulerSolver.h  -  description
                             -------------------
    begin                : 2008/04/01
    copyright            : (C) 2008 by Tomá¹ Oberhuber
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

#ifndef tnlEulerSolverH
#define tnlEulerSolverH

#include <math.h>
#include <diff/tnlExplicitSolver.h>

template< class GRID, class SCHEME, typename T = double > class tnlEulerSolver : public tnlExplicitSolver< GRID, SCHEME, T >
{
   public:

   tnlEulerSolver( const GRID& v )
   {
      k1 = new GRID( v );
      if( ! k1 )
      {
         cerr << "Unable to allocate supporting structures for the Euler solver." << endl;
         abort();
      };
      k1 -> Zeros();
   };

   tnlString GetType() const
   {
      T t;
      GRID grid;
      return tnlString( "tnlEulerSolver< " ) + grid. GetType() + 
             tnlString( ", " ) + GetParameterType( t ) + tnlString( " >" );
   };

   bool Solve( SCHEME& scheme,
               GRID& u,
               const double& stop_time,
               const double& max_res,
               const int max_iter )
   {
      T* _k1 = k1 -> Data();
      T* _u = u. Data();
           
      tnlExplicitSolver< GRID, SCHEME, T > :: iteration = 0;
      double& _time = tnlExplicitSolver< GRID, SCHEME, T > :: time;  
      double& _residue = tnlExplicitSolver< GRID, SCHEME, T > :: residue;  
      int& _iteration = tnlExplicitSolver< GRID, SCHEME, T > :: iteration;
      const double size_inv = 1.0 / ( double ) u. GetSize();
      
      T _tau = tnlExplicitSolver< GRID, SCHEME, T > :: tau;
      if( _time + _tau > stop_time ) _tau = stop_time - _time;
      if( _tau == 0.0 ) return true;

      if( tnlExplicitSolver< GRID, SCHEME, T > :: verbosity > 0 )
         tnlExplicitSolver< GRID, SCHEME, T > :: PrintOut();
      while( 1 )
      {
         int i;
         const int size = k1 -> GetSize();
         assert( size == u. GetSize() );
         
         scheme. GetExplicitRHS( _time, u, *k1 );

         double last_residue = _residue;
         double loc_residue = 0.0;
#ifdef HAVE_OPENMP
#pragma omp parallel for reduction(+:loc_residue) firstprivate( _u, _k1,_tau )
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
             :: MPIAllreduce( loc_residue, _residue, 1, MPI_SUM, tnlExplicitSolver< GRID, SCHEME, T > :: solver_comm );
         }
         _time += _tau;
         _iteration ++;
         
         if( _time + _tau > stop_time )
            _tau = stop_time - _time; //we don't want to keep such tau
         else tnlExplicitSolver< GRID, SCHEME, T > :: tau = _tau;
         
         if( tnlExplicitSolver< GRID, SCHEME, T > :: verbosity > 1 )
            tnlExplicitSolver< GRID, SCHEME, T > :: PrintOut();
         
         if( _time == stop_time || 
             ( max_res && _residue < max_res ) )
          {
            if( tnlExplicitSolver< GRID, SCHEME, T > :: verbosity > 0 )
               tnlExplicitSolver< GRID, SCHEME, T > :: PrintOut();
             return true;
          }
         //if( max_iter && _iteration == max_iter ) return false;
      }
   };

   ~tnlEulerSolver()
   {
      delete k1;
   };

   protected:
   
   GRID *k1;
};

#endif
