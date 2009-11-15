/***************************************************************************
                          mdist.cpp  -  description
                             -------------------
    begin                : 2005/08/09
    copyright            : (C) 2005 by Tom� Oberhuber
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

#include <diff/mdiff.h>

//--------------------------------------------------------------------------
m_bool GetConfiguration( mConfig& config, 
                         int argc,
                         char** argv )
{
   config. DefineCommonParams();
   config. AddParamDefinition( "epsilon", "M_REAL", "" );

   config. Set< m_int >( "mpi-split", "M_INT", 1 );
   config. Set< m_int >( "dim", "M_INT", 2 );
   config. Set< m_real >( "H", "M_REAL", 0.0 );
   config. Set< m_real >( "init-t", "M_REAL", 0.0 );
   config. Set< m_real >( "init-tau", "M_REAL", 0.0 );
   config. Set< m_real >( "final-t", "M_REAL", 0.0 );
   config. Set< m_real >( "output-period", "M_REAL", 0.0 );
   config. Set< m_real >( "adaptivity", "M_REAL", 1.0e-5 );
   config. Set< m_int >( "save-period", "M_INT", 600 );
   config. Set< mString >( "save-file", "M_STRING", "save" );
   config. Set< m_int >( "init-step", "M_INT", 0 );
   config. Set< mString >( "solver-name", "M_STRING", "merson" );
   config. Set< m_bool >( "continue", "M_BOOL", false );
   config. Set< mString >( "output-file", "M_STRING", "phi" );
   config. Set< mString >( "output-file-ending", "M_STRING", "" );

   if( ! config. ParseCLArguments( argc, argv ) )
   {
      // check at least alredy parsed arguments
      config. CheckParams();
      return false;
   }
   if( ! config. CheckParams() )
         return false;
   return true;
}
//--------------------------------------------------------------------------
int main( int argc, char* argv[] )
{
   dbgFunctionName( "", "main" );
   dbgInit( "debug.xml" );

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
   if( config. Get< m_bool >( "continue" ) )
   {
      Willmore( 0, 0, 0, 0, 0, config );
      return 0;
   }
   
   dbgCout( "Checking parameters ..." );
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
                   ! config. TestParam( "y-size", WITH_MESSAGE ) + 
   if( dim > 2 ) missing = missing +
                   ! config. TestParam( "Az", WITH_MESSAGE ) +
                   ! config. TestParam( "Bz", WITH_MESSAGE ) +
                   ! config. TestParam( "z-size", WITH_MESSAGE ) + 
   if( missing ) return -1;

   dbgCout( "Gettin parameter values ..." );
   if( dim == 1 )
   {
      m_real a = config. Get< m_real >( "Ax" );
      m_real b = config. Get< m_real >( "Bx" );
      m_real h = config. Get< m_real >( "Hx" );
      m_int size = config. Get< m_int >( "x-size" );
      mGrid1D( a, b, h, size, 0, "phi" );
      if( ! SetProblem1D( phi, config ) ) return -1;
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
   }
   if( dim == 3 )
   {
   }


}
