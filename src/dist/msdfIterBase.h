/***************************************************************************
                          msdfIterBase.h  -  description
                             -------------------
    begin                : 2008/03/13
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

#ifndef msdfIterBaseH
#define msdfIterBaseH

#include<diff/mdiff.h>

class msdfIterBase 
{
   public:

   msdfIterBase();
   
   //! Initiation
   /*! This method reads configuration parameters from mParameterContainer,
       prepares intial and initiate solver.
       All of it even for MPI.
    */
   bool InitBase( const mParameterContainer& parameters );
   
   //! Starting solver
   bool Solve();
   
   virtual void GetExplicitRHS( const double&, //time
                                mGrid2D< double >&, //_u
                                mGrid2D< double >& ) // _fu
                                {};
   virtual ~msdfIterBase();

   protected:

   //! Writes outputs like function graphs, level-set curves etc.
   bool WriteOutput( const double& time );

   bool SolveExplicit( mGrid2D< double >& u );

   //! Solution of the function
   /*! In the case of the non-MPI computing, both pointers point at the same grid.
       In the case of the MPI computing, global_u stores pointer to given global grid
       and u points to smaller grid on the subdomain belonging to the node with the rank 0.
    */
   mGrid2D< double > *u, *global_u;

   // explicit time discretisation  
   mExplicitSolver< mGrid2D< double >, msdfIterBase, double >* explicit_solver;
   
   mString output_file_format;
   mString output_file_base;
   double output_period;
   int of_stepping;
   int of_digits;
   mString log_file;
   int verbose;
   
   mString method;
   mString space_discretisation;
   mString time_discretisation;
   
   double final_time;
   double initial_tau;

   mString solver_name;
   
   double max_solver_residue;
   int max_solver_iterations;
   double merson_adaptivity;
   int gmres_restarting;
   double sor_omega;
   
   int output_index;

   bool interactive;
};

#endif
