/***************************************************************************
                          msdfIterBase.cpp  -  description
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

#include "msdfIterBase.h"

//--------------------------------------------------------------------------
msdfIterBase :: msdfIterBase()
  : u( 0 ), global_u( 0 ),
    explicit_solver( 0 ),
    output_period( 0.0 ),
    of_stepping( 0 ),
    of_digits( 0 ),
    verbose( false ),
    final_time( 0.0 ),
    initial_tau( 0.0 ),
    max_solver_residue( 0.0 ),
    max_solver_iterations( 0 ),
    merson_adaptivity( 0.0 ),
    gmres_restarting( 0.0 ),
    sor_omega( 0.0 ),
    output_index( 0.0 ),
    interactive( 0.0 )
{
}
//--------------------------------------------------------------------------
bool msdfIterBase :: InitBase( const mParameterContainer& parameters )
{
   // Getting the configuration parameters
   output_file_format = parameters. GetParameter< mString >( "output-file-format" );
   output_file_base = parameters. GetParameter< mString >( "output-file-base" );
   output_period = parameters. GetParameter< double >( "output-period" );
   of_stepping = parameters. GetParameter< int >( "of-stepping" );
   of_digits = parameters. GetParameter< int >( "of-digits" );
   if( parameters. CheckParameter( "log-file" ) )
      log_file = parameters. GetParameter< mString >( "log-file" );
   verbose = parameters. GetParameter< int >( "verbose" );
   
   //space_discretisation = parameters. GetParameter< mString >( "space-discretisation" ). Data();
   //time_discretisation = parameters. GetParameter< mString >( "time-discretisation" ). Data();
   method = parameters. GetParameter< mString >( "method" );
   if( method == "sussman-fatemi" ) time_discretisation. SetString( "explicit" );
   
   final_time = parameters. GetParameter< double >( "final-time" );
   initial_tau = parameters. GetParameter< double >( "initial-tau" );
   
   solver_name = parameters. GetParameter< mString >( "solver-name" ). Data();
   max_solver_residue = parameters. GetParameter< double >( "max-solver-residue" );
   max_solver_iterations = parameters. GetParameter< int >( "max-solver-iterations" );
   merson_adaptivity = parameters. GetParameter< double >( "merson-adaptivity" );
   gmres_restarting = parameters. GetParameter< int >( "gmres-restarting" );
   sor_omega = parameters. GetParameter< double >( "sor-omega" );
   
   interactive = parameters. GetParameter< bool >( "interactive" );
   
   return true;
}
//--------------------------------------------------------------------------
msdfIterBase :: ~msdfIterBase()
{
}
//--------------------------------------------------------------------------
bool msdfIterBase :: Solve()
{
   //if( time_discretisation == "explicit" )
      return SolveExplicit( *u );
   //if( time_discretisation == "semi-implicit" )
   //    return SolveSemiImplicit( *u );
   cerr << "Uknown method '" << time_discretisation << "' for the time discretisation." << endl;
   return false;
}
//--------------------------------------------------------------------------
bool msdfIterBase :: WriteOutput( const double& time )
{
   //mpi_mesh. Gather( *global_u, *u );
   if( ! output_file_base ) return true;
   const char* ending = "";
   if( strncmp( output_file_format. Data(), "bin", 3 ) == 0 )
      ending = ".bin";
   cout << "Writing file... " << flush;
   if( MPIGetRank() == 0 )
   {
      mString file_name;
      FileNameBaseNumberEnding(
         output_file_base. Data(),
         output_index,
         of_digits,
         ending,
         file_name );
      cout << file_name << " ... " << flush;
      if( ! Draw( *global_u, file_name. Data(), output_file_format. Data() ) )
      {
         cerr << "Sorry I could not write file " << file_name << endl;
         return false;
      }
      
      mCurve< mVector< 2, double > > curve;
      const double h = Min( global_u -> GetHx(), global_u -> GetHy() );
      GetLevelSetCurve( *global_u, curve );
      mString file_base = output_file_base + mString( "-crv" );
      FileNameBaseNumberEnding(
         file_base. Data(),
         output_index,
         of_digits,
         ending,
         file_name );
      cout << file_name << " ... " << flush;
      if( ! Write( curve, file_name. Data(), output_file_format. Data() ) )
      {
         cerr << "Sorry I could not write file " << file_name << endl;
         return false;
      }
   }
   cout << endl;
   return true;
}
//--------------------------------------------------------------------------
bool msdfIterBase :: SolveExplicit( mGrid2D< double >& _u )
{
   MPIBarrier();
   u = &_u;
   double time = 0.0;
   output_index = 0;
   
   int mpi_err( 0 );
   if( ! WriteOutput( 0 ) ) mpi_err = 1;
   :: MPIBcast( mpi_err, 1, 0 );
   if( mpi_err ) return false;
   
   const long int x_size = _u. GetXSize();
   const long int y_size = _u. GetYSize();
   long int i, j;

   assert( explicit_solver );

   explicit_solver -> SetTau( initial_tau );
   explicit_solver -> SetVerbosity( verbose );
   explicit_solver -> SetTime( time );
   cout << "Starting explicit solver..." << endl;
   while( time != final_time && time  < final_time )
   {
      double t = explicit_solver -> GetTime();
      explicit_solver -> Solve( *this, *u, Min( time + output_period, final_time ), 0, max_solver_iterations ); 
      time = explicit_solver -> GetTime();
      output_index += of_stepping;
      if( MPIGetRank() == 0 ) cout << endl;
      mpi_err = 0;
      if( ! WriteOutput( time ) ) mpi_err = 1;
      :: MPIBcast( mpi_err, 1, 0 );
      if( mpi_err ) return false;
   }
   return true;
}
