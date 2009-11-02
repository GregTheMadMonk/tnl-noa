/***************************************************************************
                          mdist.cpp  -  description
                             -------------------
    begin                : 2005/08/09
    copyright            : (C) 2005 by Tomá¹ Oberhuber
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

#include <mdiff.h>
#include "lib/iter1d.h"
#include "lib/iter2d.h"
#include "lib/iter3d.h"
#include "lib/fmm2d.h"
#include "lib/fsm2d.h"
#include "lib/ftm2d.h"
#include "lib/debug.h"

m_bool CheckParameters( mConfig& config )
{
   
   m_int dim;
   if( ! config. TestParam( "dim" ) )
   {
      cerr << "Missing parameter 'dim'." << endl;
      return false;
   }
   dim = config. Get< m_int >( "dim" );
  
   if( dim < 1 && dim > 3 )
   {
      cerr << "Problem dimension can be only 1, 2 or 3. " << endl;
      return false;
   }

   m_char* param_data[ 5 ][ 5 ] =
   { { "M_REAL", "A", "Ax", "Ay", "Az" },
     { "M_REAL", "B", "Bx", "By", "Bz" },
     { "M_INT", "mpi-split", "mpi-x-split", "mpi-y-split", "mpi-z-split"},
     { "M_REAL", "H", "Hx", "Hy", "Hz" },
     { "M_INT", "size", "x-size", "y-size", "z-size" } };

   m_int i, j;

   for( i = 0; i < 5; i ++ )
   {
      DBG_EXPR( param_data[ i ][ 1 ] );
      if( config. TestParam( param_data[ i ][ 1 ] ) )
         for( j = 2; j < 5; j ++ )
         {
            DBG_EXPR( param_data[ i ][ j ] );
            if( ! config. TestParam( param_data[ i ][ j ] ) )
               config. SetParam( param_data[ i ][ j ],
                                 param_data[ i ][ 0 ],
                                 config. GetParam( param_data[ i ][ 1 ] ) );
         }
   }

   if( ! config. TestParam( "init-time" ) )
      config. Set< m_real >( "init-time", "M_REAL", 0.0 );
   return true;

}
//--------------------------------------------------------------------------
m_bool GetConfiguration( mConfig& config, 
                         int argc,
                         char** argv )
{
   config. AddClassDefinition( "IO", "Input and output data" );
   config. AddClassDefinition( "METHOD", "Method setting" );
   config. AddClassDefinition( "SOLVER", "Solver setting" );
   config. AddClassDefinition( "PROBLEM", "Problem settings" );
   config. AddClassDefinition( "BND", "Boundary conditions" );

   //************************ IO **************************************
   config. AddParamDefinition( "output-period",
                       "M_REAL",
                       "Time step for outputs",
                       "IO" ); 
   config. AddParamDefinition( "output-file",
                       "M_STRING",
                       "Base for output file name. An index of output will be added.",
                       "IO" ); 
   config. AddParamDefinition( "output-file-ending",
                       "M_STRING",
                       "Ending for the outpu file name.",
                       "IO" ); 
   config. AddParamDefinition( "input-file",
                       "M_STRING",
                       "Input file for initial condition",
                       "IO" ); 
   config. AddParamDefinition( "log-file",
                       "M_STRING",
                       "Log file name.",
                       "IO" ); 
   config. AddParamDefinition( "save-file",
                       "M_STRING",
                       "File name for periodical saving of the computation state. Useful for the case of accidental interuption.",
                       "IO" ); 
   config. AddParamDefinition( "save-file-ending",
                       "M_STRING",
                       "Ending for the computation save file.",
                       "IO" ); 
   config. AddParamDefinition( "save-period",
                       "M_INT",
                       "Period in seconds for saving the state of the computation.",
                       "IO" ); 
   config. AddParamDefinition( "continue",
                       "M_BOOL",
                       "Computation restoration after interuption usin the computation save file",
                       "IO" ); 
   config. AddParamDefinition( "verbose",
                       "M_INT",
                       "Verbose mode.",
                       "IO" ); 

   //******************* Problem settings ******************************
   config. AddParamDefinition( "problem-name", "M_STRING", "Set problem name.", "PROBLEM" ); 
   config. AddParamDefinition( "dim", "M_INT", "Problem dimension.", "PROBLEM" ); 
   config. AddParamDefinition( "size", "M_INT", "Grid size.", "PROBLEM" ); 
   config. AddParamDefinition( "x-size", "M_INT", "", "PROBLEM" ); 
   config. AddParamDefinition( "y-size", "M_INT", "", "PROBLEM" ); 
   config. AddParamDefinition( "z-size", "M_INT", "", "PROBLEM" ); 
   config. AddParamDefinition( "A", "M_REAL", "Left-bottom corner of the problem domain.", "PROBLEM" ); 
   config. AddParamDefinition( "Ax", "M_REAL", "", "PROBLEM" ); 
   config. AddParamDefinition( "Ay", "M_REAL", "", "PROBLEM" ); 
   config. AddParamDefinition( "Az", "M_REAL", "", "PROBLEM" ); 
   config. AddParamDefinition( "B", "M_REAL", "Right-top corner of the problem domain.", "PROBLEM" ); 
   config. AddParamDefinition( "Bx", "M_REAL", "", "PROBLEM" ); 
   config. AddParamDefinition( "By", "M_REAL", "", "PROBLEM" ); 
   config. AddParamDefinition( "Bz", "M_REAL", "", "PROBLEM" ); 
   config. AddParamDefinition( "H", "M_REAL", "Mesh space step.", "PROBLEM" ); 
   config. AddParamDefinition( "Hx", "M_REAL", "", "PROBLEM" ); 
   config. AddParamDefinition( "Hy", "M_REAL", "", "PROBLEM" ); 
   config. AddParamDefinition( "Hz", "M_REAL", "", "PROBLEM" ); 
    

   config. AddParamDefinition( "epsilon", "M_REAL", "" );
   config. AddParamDefinition( "ampl", "M_REAL", "" );
   config. AddParamDefinition( "freq", "M_REAL", "" );
   config. AddParamDefinition( "radius", "M_REAL", "" );

   config. Set< m_int >( "mpi-split", "M_INT", 1 );
   config. Set< m_int >( "dim", "M_INT", 2 );
   config. Set< m_real >( "H", "M_REAL", 0.0 );
   config. Set< m_real >( "init-t", "M_REAL", 0.0 );
   config. Set< m_real >( "init-tau", "M_REAL", 0.0 );
   config. Set< m_real >( "final-t", "M_REAL", 0.0 );
   config. Set< m_real >( "output-period", "M_REAL", 0.0 );
   config. Set< m_real >( "adaptivity", "M_REAL", 1.0e-3 );
   config. Set< m_int >( "save-period", "M_INT", 600 );
   config. Set< mString >( "save-file", "M_STRING", "save" );
   config. Set< m_int >( "init-step", "M_INT", 0 );
   config. Set< mString >( "solver-name", "M_STRING", "merson" );
   config. Set< m_bool >( "continue", "M_BOOL", false );
   config. Set< mString >( "output-file", "M_STRING", "phi" );
   config. Set< mString >( "output-file-ending", "M_STRING", "" );
   config. Set< m_real >( "epsilon", "M_REAL", 1.0 );
   config. Set< m_real >( "radius", "M_REAL", 1.0 );

   if( ! config. ParseCLArguments( argc, argv ) )
   {
      // check at least alredy parsed arguments
      CheckParameters( config );
      return false;
   }
   if( ! CheckParameters( config ) )
         return false;
   return true;
}
//--------------------------------------------------------------------------
m_bool SetProblem1D( mGrid1D& phi, const mConfig& config )
{
   m_int x_size = phi. Size();
   m_real a = phi. A();
   m_real h = phi. H();
   const m_char* problem_name = 
      config. Get< mString >( "problem-name" ). Data();
   if( strcmp( problem_name, "sin-waves" ) == 0 )
   {
      if( ! config. TestParam( "ampl", WITH_MESSAGE ) ||
          ! config. TestParam( "freq", WITH_MESSAGE ) )
         return false;
      m_real ampl = config. Get< m_real >( "ampl" );
      m_real freq = config. Get< m_real >( "freq" );
      m_int i;
      for( i = 0; i < x_size; i ++ )
         phi( i ) = ampl * sin( freq * M_PI * ( a + i * h ) );
      return true;
   }
   return false;
}
//--------------------------------------------------------------------------
m_bool SetProblem2D( mGrid2D& phi, const mConfig& config )
{
   m_int x_size = phi. XSize();
   m_int y_size = phi. YSize();
   m_real a_x = phi. A(). x;
   m_real a_y = phi. A(). y;
   m_real h_x = phi. H(). x;
   m_real h_y = phi. H(). y;
   const m_char* problem_name = 
      config. Get< mString >( "problem-name" ). Data();
   if( strcmp( problem_name, "circle" ) == 0 )
   {
      m_real radius = config. Get< m_real >( "radius" );
      m_int i, j;
      for( i = 0; i < x_size; i ++ )
         for( j = 0; j < y_size; j ++ )
         {
            m_real x = a_x + i * h_x;
            m_real y = a_y + j * h_y;
            phi( i, j ) = x * x + y * y - radius * radius;
         }
      return true;
   }
   if( strcmp( problem_name, "sin-waves" ) == 0 )
   {
      if( ! config. TestParam( "ampl", WITH_MESSAGE ) ||
          ! config. TestParam( "freq", WITH_MESSAGE ) )
         return false;
      m_real ampl = config. Get< m_real >( "ampl" );
      m_real freq = config. Get< m_real >( "freq" );
      m_int i, j;
      for( i = 0; i < x_size; i ++ )
         for( j = 0; j < y_size; j ++ )
         {
            m_real x = a_x + i * h_x;
            m_real y = a_y + j * h_y;
            phi( i, j ) = ampl * 
               sin( freq * M_PI * ( sqrt( x * x + y * y ) ) );
         }
      return true;
   }
   if( strcmp( problem_name, "checkboard" ) == 0 )
   {
      if( ! config. TestParam( "ampl", WITH_MESSAGE ) ||
          ! config. TestParam( "freq", WITH_MESSAGE ) )
         return false;
      m_real ampl = config. Get< m_real >( "ampl" );
      m_real freq = config. Get< m_real >( "freq" );
      m_int i, j;
      for( i = 0; i < x_size; i ++ )
         for( j = 0; j < y_size; j ++ )
         {
            m_real x = a_x + i * h_x;
            m_real y = a_y + j * h_y;
            phi( i, j ) = ampl * 
               sin( freq * M_PI * x ) *
               sin( freq * M_PI * y );
         }
      return true;
   }
   return false;
}
//--------------------------------------------------------------------------
int main( int argc, char* argv[] )
{
   DBG_FUNCTION_NAME( "", "main" );
   DBG_INIT( "debug.xml" );

   mConfig config;
   
   /*MPIInit( &argc, &argv );
   mMPIMesh2D mpi_mesh;
   if( mpi_mesh. NodeNumber() != 0 )
   {
      Willmore( 0, 0, 0, 0, 0, config );
      return 0;
   }*/
   
   if( ! GetConfiguration( config, argc, argv ) )
      return -1;
   /*if( config. Get< m_bool >( "continue" ) )
   {
      Willmore( 0, 0, 0, 0, 0, config );
      return 0;
   }*/
   
   DBG_COUT( "Checking parameters ..." );
   if( ! config. TestParam( "dim", WITH_MESSAGE ) )
      return -1;
   m_int dim = config. Get< m_int >( "dim" );
   m_int missing = ! config. TestParam( "Ax", WITH_MESSAGE ) +
                   ! config. TestParam( "Bx", WITH_MESSAGE ) +
                   ! config. TestParam( "x-size", WITH_MESSAGE ) +
                   ! config. TestParam( "problem-name", WITH_MESSAGE );
   if( dim > 1 ) missing = missing +
                   ! config. TestParam( "Ay", WITH_MESSAGE ) +
                   ! config. TestParam( "By", WITH_MESSAGE ) +
                   ! config. TestParam( "y-size", WITH_MESSAGE );
   if( dim > 2 ) missing = missing +
                   ! config. TestParam( "Az", WITH_MESSAGE ) +
                   ! config. TestParam( "Bz", WITH_MESSAGE ) +
                   ! config. TestParam( "z-size", WITH_MESSAGE );
   if( missing ) return -1;

   const m_char* method = config. Get< mString >( "method-name" ). Data();
   DBG_COUT( "Gettin parameter values ..." );
   if( dim == 1 )
   {
      m_real a = config. Get< m_real >( "Ax" );
      m_real b = config. Get< m_real >( "Bx" );
      m_real h = config. Get< m_real >( "Hx" );
      m_int size = config. Get< m_int >( "x-size" );
      mGrid1D phi( a, b, h, size, 0, "phi" );
      if( ! SetProblem1D( phi, config ) ) return -1;

      IterDist1D( &phi, config );
   }
   if( dim == 2 )
   {
      mVector2D A( config. Get< m_real >( "Ax" ),
                   config. Get< m_real >( "Ay" ) );
      mVector2D B( config. Get< m_real >( "Bx" ),
                   config. Get< m_real >( "By" ) );
      mVector2D H( config. Get< m_real >( "Hx" ),
                   config. Get< m_real >( "Hy" ) );
      m_int x_size = config. Get< m_int >( "x-size" );
      m_int y_size = config. Get< m_int >( "y-size" );
      
      mGrid2D phi( A, B, H, x_size, y_size, 0, 0, "phi" );
      if( ! SetProblem2D( phi, config ) ) return -1;
      
      mCurve2D curve;
      curve. Identify( phi );
      curve. Draw( "init_curve" );
      
      if( strcmp( method, "fast-marching" ) == 0 )
      {
         DstFastMarching2D( phi, 0.0, 0.0 );
         cout << "Writing file phi_dst..." << endl;
         phi. DrawFunction( "phi_dst" );
         return 0;
      }
      if( strcmp( method, "fast-sweeping" ) == 0 )
      {
         for( int i = 0; i < 1000; i ++ )
           DstFastSweeping2D( phi, 4, 0, MDST_VERBOSE_ON );
         cout << "Writing file phi_dst..." << endl;
         phi. DrawFunction( "phi_dst" );
         curve. Identify( phi );
         curve. Draw( "final_curve" );
         return 0;
      }
      if( strcmp( method, "front-tracing" ) == 0 )
      {
         DstFrontTracing2D( phi, 0.0 );
         cout << "Writing file phi_dst..." << endl;
         phi. DrawFunction( "phi_dst" );
         return 0;
      }

      IterDist2D( &phi, config );
   }
   if( dim == 3 )
   {
   }


}
