/***************************************************************************
                          mdist-test-param-crv-dst.cpp  -  description
                             -------------------
    begin                : 2007/02/24
    copyright            : (C) 2007 by Tomá¹ Oberhuber
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

//--------------------------------------------------------------------------
#include <sys/utsname.h>
#include <time.h>
#include <mdiff.h>
#include "debug.h"
#include "mdist.h"
#include "sdf-restore-def.h"
#include "initial-condition.h"
#include "msdfSussmanFatemi.h"

//--------------------------------------------------------------------------
void WriteProlog( ostream& str, 
                  const mParameterContainer& parameters,
                  const double& hx,
                  const double& hy,
                  const long int xsize,
                  const long int ysize )
{
   mLogger logger( 70, str ); 
   logger. WriteHeader( "SDF-Restore" );
   logger. WriteParameter< mString >( "Method:", "method", parameters );
   const mString initial_condition = parameters. GetParameter< mString >( "initial-condition" );
   logger. WriteParameter< mString >( "Initial condition:", "initial-condition", parameters );
   logger. WriteParameter< bool >( "Conevergence test:", "convergence-test", parameters, 1 );
   if( initial_condition == "sign-hole" )
      logger. WriteParameter< double >( "Radius:", "radius", parameters, 1 );
   if( initial_condition == "wavy-hole" ||
       initial_condition == "wavy-circles" )
   {
      logger. WriteParameter< double >( "Radius:", "radius", parameters, 1 );
      logger. WriteParameter< double >( "Frequency:", "frequency", parameters, 1 );
      logger. WriteParameter< double >( "Amplitude:", "amplitude", parameters, 1 );
      logger. WriteParameter< double >( "Phase:", "phase", parameters, 1 );
   }
   if( initial_condition == "exp-square" )
   {
      logger. WriteParameter< double >( "Power:", "power", parameters, 1 );
      logger. WriteParameter< double >( "Sigma:", "sigma", parameters, 1 );
   }
   if( initial_condition == "sin-waves" ||
       initial_condition == "sin-waves-neumann" )
   {
      logger. WriteParameter< double >( "Frequency:", "frequency", parameters, 1 );
      logger. WriteParameter< double >( "Amplitude:", "amplitude", parameters, 1 );
   }
   if( initial_condition == "wavy-band" ||
       initial_condition == "wavy-band-1-1" ||
       initial_condition == "wavy-band-1--1" )
   {
      logger. WriteParameter< double >( "Frequency:", "frequency", parameters, 1 );
      logger. WriteParameter< double >( "Amplitude:", "amplitude", parameters, 1 );
   }
   if( initial_condition == "sphere-restore" ||
       initial_condition == "perturbed-sphere" )
   {
      logger. WriteParameter< double >( "Frequency:", "frequency", parameters, 1 );
      logger. WriteParameter< double >( "Amplitude:", "amplitude", parameters, 1 );
      logger. WriteParameter< double >( "Sigma:", "sigma", parameters, 1 );
      logger. WriteParameter< double >( "Shift:", "shift", parameters, 1 );
   }
   if( initial_condition == "circle" ||
       initial_condition == "astroid" ||
       initial_condition == "two-circles" ||
       initial_condition == "four-circles" ||
       initial_condition == "five-circles" )
      logger. WriteParameter< double >( "Radius:", "radius", parameters, 1 );
   if( initial_condition == "ellipse" )
   {
      logger. WriteParameter< double >( "Radius-1:", "radius1", parameters, 1 );
      logger. WriteParameter< double >( "Radius-2:", "radius2", parameters, 1 );
   }
   if( initial_condition == "ellipse-circle-test" )
   {
      logger. WriteParameter< double >( "Circle radius:", "radius", parameters, 1 );
      logger. WriteParameter< double >( "Ellipse radius-1:", "radius1", parameters, 1 );
      logger. WriteParameter< double >( "Ellipse radius-2:", "radius2", parameters, 1 );
   }
   if( initial_condition == "michal" ||
       initial_condition == "flower" ||
       initial_condition == "star" )
   {
      logger. WriteParameter< double >( "Radius:", "radius", parameters, 1 );
      logger. WriteParameter< double >( "Amplitude:", "amplitude", parameters, 1 );
      logger. WriteParameter< double >( "Frequency:", "frequency", parameters, 1 );
   }
   logger. WriteSeparator();
   //const mString& space_discretisation = parameters. GetParameter< mString >( "space-discretisation" );
   //logger. WriteParameter< mString >( "Space discretisation:", "space-discretisation", parameters );
   logger. WriteParameter< double >( "Domain left side:", "Ax", parameters, 1 );
   logger. WriteParameter< double >( "Domain right side:", "Bx", parameters, 1 );
   logger. WriteParameter< double >( "Domain bottom side:", "Ay", parameters, 1 );
   logger. WriteParameter< double >( "Domain top side:", "By", parameters,1 );
   logger. WriteParameter< double >( "X space step:", hx, 1 );
   logger. WriteParameter< double >( "Y space step:", hy, 1 );
   logger. WriteParameter< int >( "X grid dimension:", xsize, 1 );
   logger. WriteParameter< int >( "Y grid dimension:", ysize, 1 );
   //logger. WriteParameter< mString >( "Time discretisation:", "time-discretisation", parameters );
   logger. WriteParameter< double >( "Initial tau:", "initial-tau", parameters, 1 );
   logger. WriteParameter< double >( "Final time:", "final-time", parameters, 1 );
   logger. WriteSeparator();
   const mString& solver_name = parameters. GetParameter< mString >( "solver-name" );
   logger. WriteParameter< mString >( "Solver:", "solver-name", parameters );
   if( solver_name == "merson" )
      logger. WriteParameter< double >( "Adaptivity:", "merson-adaptivity", parameters, 1 );
   if( solver_name == "sor" )
      logger. WriteParameter< double >( "Omega:", "sor-omega", parameters, 1 );
   if( solver_name == "gmres" )
      logger. WriteParameter< int >( "Restarting:", "gmres-restarting", parameters, 1 );
   if( parameters. CheckParameter( "preconditioner" ) )
      logger. WriteParameter< mString >( "Preconditioner:", "preconditioner", parameters, 1 ); 
   logger. WriteParameter< int >( "Maximum of iterations:", "max-solver-iterations", parameters );
   logger. WriteParameter< double >( "Maximal residue:", "max-solver-residue", parameters );
   logger. WriteSeparator();
   logger. WriteParameter< mString >( "Output file name:", "output-file-base", parameters );
   logger. WriteParameter< mString >( "Output file format:", "output-file-format", parameters );
   logger. WriteParameter< double >( "Output period:", "output-period", parameters );
   logger. WriteSeparator();
   char host_name[ 256 ];
   struct utsname uts;
   gethostname( host_name, 255 );
   uname( &uts );
   logger. WriteParameter< char* >( "Host name:", host_name );
   logger. WriteParameter< char* >( "Architecture:", uts. machine );
   fstream file;
   file. open( "/proc/cpuinfo", ios :: in );
   if( file )
   {
      char line[ 1024 ];
      char* cpu_id;
      char* cpu_model_name;
      char* cpu_mhz;
      char* cpu_cache;
      while( ! file. eof() )
      {
         int i;
         file. getline( line, 1024 );
         if( strncmp( line, "processor", strlen( "processor" ) ) == 0 )
         {
            i = strlen( "processor" );
            while( line[ i ] != ':' && line[ i ] ) i ++;
            cpu_id = &line[ i + 1 ];
            logger. WriteParameter< char * >( "CPU Id.:", cpu_id );
            continue;
         }
         if( strncmp( line, "model name", strlen( "model name" ) ) == 0 )
         {
            i = strlen( "model name" );
            while( line[ i ] != ':' && line[ i ] ) i ++;
            cpu_model_name = &line[ i + 1 ];
            logger. WriteParameter< char * >( "Model name:", cpu_model_name );
            continue;
         }
         if( strncmp( line, "cpu MHz", strlen( "cpu MHz" ) ) == 0 )
         {
            i = strlen( "cpu MHz" );
            while( line[ i ] != ':' && line[ i ] ) i ++;
            cpu_mhz = &line[ i + 1 ];
            logger. WriteParameter< char * >( "CPU MHz:", cpu_mhz );
            continue;
         }
         if( strncmp( line, "cache size", strlen( "cache size" ) ) == 0 )
         {
            i = strlen( "cache size" );
            while( line[ i ] != ':' && line[ i ] ) i ++;
            cpu_cache = &line[ i + 1 ];
            logger. WriteParameter< char * >( "CPU cache:", cpu_cache );
            continue;
         }
      }
   }
   file. close();
   logger. WriteParameter< char* >( "System:", uts. sysname );
   logger. WriteParameter< char* >( "Release:", uts. release );
   logger. WriteParameter< char* >( "Compiler:", CPP_COMPILER_NAME );
   logger. WriteSeparator();
   time_t timeval;
   time( &timeval );
   tm *tm_ptr = localtime( &timeval );
   char buf[ 256 ];
   strftime( buf, 256, "%a %b %d %H:%M:%S\0", tm_ptr );
   logger. WriteParameter< char* >( "Started at:", buf );
}
//--------------------------------------------------------------------------
void WriteEpilog( ostream& str )
{
   mLogger logger( 70, str ); 
   time_t timeval;
   time( &timeval );
   tm *tm_ptr = localtime( &timeval );
   char buf[ 256 ];
   strftime( buf, 256, "%a %b %d %H:%M:%S\0", tm_ptr );
   logger. WriteParameter< char* >( "Finished at:", buf );
   int cpu_time = default_mcore_cpu_timer. GetTime();
   sprintf( buf, "%d sec", cpu_time );
   logger. WriteParameter< char* >( "CPU Time:", buf );
   double rt_time = default_mcore_rt_timer. GetTime();
   sprintf( buf, "%f sec", rt_time );
   logger. WriteParameter< char* >( "Real Time:", buf );
   sprintf( buf, "%f %%", 100 * ( ( double ) cpu_time ) / rt_time );
   logger. WriteParameter< char* >( "CPU usage:", buf );
   logger. WriteSeparator();
}
//-------------------------------------------------------------------------- 
int main( int argc, char* argv[] )
{
   DBG_FUNCTION_NAME( "", "main" );
   DBG_INIT( "debug.xml" );
   
   mParameterContainer parameters;
   mConfigDescription conf_desc;
   if( conf_desc. ParseConfigDescription( CONFIG_DESCRIPTION_FILE ) != 0 )
      return 1;
   if( ! ParseCommandLine( argc, argv, conf_desc, parameters ) )
   {
      conf_desc. PrintUsage( argv[ 0 ] );
      return 1;
   }
      
   long int x_size, y_size;
   if( ! parameters. CheckParameter( "x-size" ) ||
       ! parameters. CheckParameter( "y-size" ) ) 
      if( ! parameters. CheckParameter( "size" ) )
      {
         cerr << "Missing either parameter size or x-size together with y-size." << endl;
         return 1;
      } 
      else x_size = y_size = parameters. GetParameter< int >( "size" );
   else
   {
      x_size = parameters. GetParameter< int >( "x-size" );
      y_size = parameters. GetParameter< int >( "y-size" );
   }
   double Ax, Ay, Bx, By;
   Ax = parameters. GetParameter< double >( "Ax" );
   Ay = parameters. GetParameter< double >( "Ay" );
   Bx = parameters. GetParameter< double >( "Bx" );
   By = parameters. GetParameter< double >( "By" );
   
   mGrid2D< double >* u = new mGrid2D< double >( x_size, y_size, Ax, Bx, Ay, By );
   
   cout << "Setting up the initial condition..." << endl;
   if( ! GetInitialCondition( parameters, u ) ) 
      return 1;

   mString method = parameters. GetParameter< mString >( "method" );
   const double hx = u -> GetHx();
   const double hy = u -> GetHy();
   
   if( MPIGetRank() == 0 )
   {
      WriteProlog( cout,
                   parameters,
                   hx, hy, x_size, y_size );
      mString log_file_name;
      if( parameters. GetParameter< mString >( "log-file", log_file_name ) )
      {
         fstream file;
         file. open( log_file_name. Data(), ios :: out );
         if( ! file )
         {
            cerr << "Sorry I can not open the file " << log_file_name << " for writing log." << endl;
            return 1;
         }
         WriteProlog( file,
                      parameters,
                      hx, hy, x_size, y_size );
         file. close();
      }
   }

   bool error( false );
   if( method == "sussman-fatemi" )
   {
      msdfSussmanFatemi scheme;      
      cout << "Initiating the Sussman-Fatemi scheme..." << endl;
      if( ! scheme. Init( parameters, u ) ) error = true;
      else
      {
         cout << "Starting solver..." << endl;
         scheme. Solve();
      }
   }
   
   if( MPIGetRank() == 0 )
   {
      cout << endl;
      WriteEpilog( cout );
      mString log_file_name;
      if( parameters. GetParameter< mString >( "log-file", log_file_name ) )
      {
         fstream file;
         file. open( log_file_name. Data(), ios :: out | ios :: app );
         if( ! file )
         {
            cerr << "Sorry I can not open the file " << log_file_name << " for writing log." << endl;
            return 1;
         }
         cout << "Writing to " << log_file_name << endl;
         WriteEpilog( file );
         file. close();
      }
      delete u;
   }
   MPIFinalize();
   
   if( error ) return false;
   return true;

}
