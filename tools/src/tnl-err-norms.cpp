/***************************************************************************
                          diff-norm.cpp  -  description
                             -------------------
    begin                : 2007/07/05
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

#include <fstream>
#include <stdio.h>
#include <math.h>

#include <debug/tnlDebug.h>
#include <mesh/tnlGrid.h>
#include <config/tnlConfigDescription.h>
#include <config/tnlParameterContainer.h>
#include <core/tnlCurve.h>

#include "../../mdiff-err-norms-def.h"

#include "tnl-err-norms.h"

void WriteHeader( ostream& stream,
                  const double& tau )
{
   stream << setw( 8 ) << "N";
   if( tau ) stream << setw( 12 ) << "time";
   stream << setw( 12 ) << "h"
          << setw( 15 ) << "L1 - norm" 
          << setw( 15 ) << "L2 - norm"
          << setw( 15 ) << "MAX - norm" << endl;
}

void WriteLine( ostream& stream,
                const int N,
                const double& tau,
                const double& h,
                const double& l1_norm,
                const double& l2_norm,
                const double& max_norm )
{
   stream << setw( 8 ) << N;
   if( tau ) stream << setw( 12 ) << N * tau;
   stream << setw( 12 ) << h 
          << setw( 15 ) << l1_norm
          << setw( 15 ) << l2_norm
          << setw( 15 ) << max_norm << endl;
}

void WriteLastLine( ostream& stream,
                    const double& h,
                    const double& l1_int,
                    const double& l2_int,
                    const double& max_int )
{
   stream << setw( 8 ) << "Total:"
          << setw( 12 ) << ""
          << setw( 12 ) << h
          << setw( 15 ) << l1_int
          << setw( 15 ) << l2_int
          << setw( 15 ) << max_int << endl << endl;
}

void WriteGraphLine( ostream& stream,
                     const int N,
                     const double& tau,
                     const double& l1_norm,
                     const double& l2_norm,
                     const double& max_norm )
{
   if( tau ) stream << setw( 12 ) << N * tau;
   stream << setw( 15 ) << l1_norm
          << setw( 15 ) << l2_norm
          << setw( 15 ) << max_norm << endl;
}

void WriteLogGraphLine( ostream& stream,
                     const int N,
                     const double& tau,
                     const double& l1_norm,
                     const double& l2_norm,
                     const double& max_norm )
{
   if( tau ) stream << setw( 12 ) << N * tau;
   stream << setw( 15 ) << log10( l1_norm )
          << setw( 15 ) << log10( l2_norm )
          << setw( 15 ) << log10( max_norm ) << endl;
}

int main( int argc, char* argv[] )
{
   dbgFunctionName( "", "main" );
   dbgInit( "mdiff-err-norms.dbg" );
   tnlParameterContainer parameters;
   tnlConfigDescription conf_desc;

   dbgCout( "Parsing config description file ... " );
   if( conf_desc. ParseConfigDescription( CONFIG_DESCRIPTION_FILE ) != 0 )
      return 1;
   if( ! ParseCommandLine( argc, argv, conf_desc, parameters ) )
   {
      conf_desc. PrintUsage( argv[ 0 ] );
      return 1;
   }
   int verbose = parameters. GetParameter< int >( "verbose" );
   if( verbose )
   {
      cout << endl;
      cout << "************************************************************************" << endl;
      cout << "*                                                                      *" << endl;
      cout << "*           TNL Tools: grid difference                                 *" << endl;
      cout << "*                                                                      *" << endl;
      cout << "************************************************************************" << endl;
   }

   tnlString test = parameters. GetParameter< tnlString >( "test" );
   tnlList< tnlString > first_files = parameters. GetParameter< tnlList< tnlString > >( "first-set" );
   tnlList< tnlString > second_files;
   if( test == "function" )
   {
      if( ! parameters. CheckParameter( "second-set" ) )
      {
         cerr << "Missing parameter second-set." << endl;
         return 1;
      }
      second_files = parameters. GetParameter< tnlList< tnlString > >( "second-set" );
   }
   int size = first_files. getSize();
   if( verbose )
      cout << "Processing " << size << " files. " << endl;

   int edge_skip = parameters. GetParameter< int >( "edges-skip" );
   bool write_difference = parameters. GetParameter< bool >( "write-difference" );
   bool write_exact_curve = parameters. GetParameter< bool >( "write-exact-curve" );
   bool write_graph = parameters. GetParameter< bool >( "write-graph" );
   bool write_log_graph = parameters. GetParameter< bool >( "write-log-graph" );
   int first2second_ratio = parameters. GetParameter< int >( "first-to-second-ratio" );
   double space_step = parameters. GetParameter< double >( "space-step" );
   double tau( 0.0 );
   parameters. GetParameter< double >( "tau", tau );
   tnlString output_file_name;
   fstream output_file;
   if( parameters. GetParameter< tnlString >( "output-file", output_file_name ) )
   {
      // Open the output file
      output_file. open( output_file_name. getString(), ios :: out );
      if( ! output_file )
      {
         cerr << "Sorry I am not able to open the file " << output_file_name << "." << endl;
         return 1;
      }
      
   }
   else WriteHeader( cout, tau );
   fstream graph_file;
   if( write_graph )
   {
      graph_file. open( "graph.gplt", ios :: out );  
      if( ! graph_file )
      {
         cerr << "Sorry I am not able to open the file graph.gplt." << endl;
         return 1;
      }
   }
   fstream log_graph_file;
   if( write_log_graph )
   {
      log_graph_file. open( "log-graph.gplt", ios :: out );  
      if( ! log_graph_file )
      {
         cerr << "Sorry I am not able to open the file log-graph.gplt." << endl;
         return 1;
      }
   }
   
   
   double l1_int( 0.0 ), l2_int( 0.0 ), max_int( 0.0 ), h( 0.0 );
   if( test == "function" )
   {
      if( size / first2second_ratio != second_files. getSize() && second_files. getSize() != 1 )
      {
         cerr << "Sorry, the number of the files in the second set does not agree with the number of the files in the second one!" << endl;
         return 1;
      }
      for( int i = 0; i < size; i ++ )
      {
         tnlString firstFileName, secondFileName;
         firstFileName = first_files[ i ];
         if( verbose )
            cout << "Processing the file " << firstFileName << " ...          \r" << flush;
         tnlString firstObjectType, secondObjectType;
         if( ! getObjectType( first_files[ i ], firstObjectType ) )
            return EXIT_FAILURE;

         if( i % first2second_ratio == 0 && i / first2second_ratio < second_files. getSize() )
         {
            secondFileName = second_files[ i / first2second_ratio ];
            if( verbose )
               cout << endl << "Comparing with the file " << secondFileName << " ...          \r" << flush;
            if( ! getObjectType( secondFileName, secondObjectType ) )
               return EXIT_FAILURE;
         }

         double l1_norm( 0.0 ), l2_norm( 0.0 ), max_norm( 0.0 );
         bool objects_match( false );
         if( ( firstObjectType == "tnlGrid< 2, double, tnlHost, int >" ||
               firstObjectType == "tnlGrid< 2, double, tnlCuda, int >") &&
             ( secondObjectType == "tnlGrid< 2, double, tnlHost, int >" ||
               secondObjectType == "tnlGrid< 2, double, tnlCuda, int >" ) )
         {
            compareGrids< 2, double, double, tnlHost, tnlHost, int >( firstFileName,
                                                                      secondFileName,
                                                                      write_difference,
                                                                      edge_skip,
                                                                      space_step,
                                                                      l1_norm,
                                                                      l2_norm,
                                                                      max_norm,
                                                                      h,
                                                                      verbose );
            objects_match = true;
         }
         if( ( firstObjectType == "tnlGrid< 2, float, tnlHost, int >" ||
               firstObjectType == "tnlGrid< 2, float, tnlCuda, int >" ) &&
             ( secondObjectType == "tnlGrid< 2, float, tnlHost, int >" ||
               secondObjectType == "tnlGrid< 2, float, tnlCuda, int >" ) )
         {
            float _l1_norm, _l2_norm, _max_norm;
            float _space_step = ( float ) space_step;
            float _h = ( float ) h;
            compareGrids< 2, float, float, tnlHost, tnlHost, int >( firstFileName,
                                                                    secondFileName,
                                                                    write_difference,
                                                                    edge_skip,
                                                                    _space_step,
                                                                    _l1_norm,
                                                                    _l2_norm,
                                                                    _max_norm,
                                                                    _h,
                                                                    verbose );
            l1_norm = ( double ) _l1_norm;
            l2_norm = ( double ) _l2_norm;
            max_norm = ( double ) _max_norm;
            h = ( double ) _h;
            objects_match = true;
         }
         if( ( firstObjectType == "tnlGrid< 2, double, tnlHost, int >" ||
               firstObjectType == "tnlGrid< 2, double, tnlCuda, int >" ) &&
             ( secondObjectType == "tnlGrid< 2, float, tnlHost, int >" ||
               secondObjectType == "tnlGrid< 2, float, tnlCuda, int >" ) )
         {
            compareGrids< 2, double, float, tnlHost, tnlHost, int >( firstFileName,
                                                                     secondFileName,
                                                                     write_difference,
                                                                     edge_skip,
                                                                     space_step,
                                                                     l1_norm,
                                                                     l2_norm,
                                                                     max_norm,
                                                                     h,
                                                                     verbose );
            objects_match = true;
         }
         if( ( firstObjectType == "tnlGrid< 2, float, tnlHost, int >" ||
               firstObjectType == "tnlGrid< 2, float, tnlCuda, int >" ) &&
             ( secondObjectType == "tnlGrid< 2, double, tnlHost, int >" ||
               secondObjectType == "tnlGrid< 2, double, tnlHost, int >" ) )
         {
            compareGrids< 2, double, float, tnlHost, tnlHost, int >( secondFileName,
                                                                     firstFileName,
                                                                     write_difference,
                                                                     edge_skip,
                                                                     space_step,
                                                                     l1_norm,
                                                                     l2_norm,
                                                                     max_norm,
                                                                     h,
                                                                     verbose );
            objects_match = true;
         }
         if( firstObjectType == "tnlGrid< 2, tnlDouble, tnlHost, int >" ||
             firstObjectType == "tnlGrid< 2, tnlDouble, tnlCuda, int >" )
         {
            tnlDouble _l1_norm, _l2_norm, _max_norm;
            tnlDouble _space_step = ( tnlDouble ) space_step;
            tnlDouble _h = ( tnlDouble ) h;

            compareGrids< 2, tnlDouble, tnlDouble, tnlHost, tnlHost, int >( firstFileName,
                                                                            secondFileName,
                                                                            write_difference,
                                                                            edge_skip,
                                                                            _space_step,
                                                                            _l1_norm,
                                                                            _l2_norm,
                                                                            _max_norm,
                                                                            _h,
                                                                            verbose );
            l1_norm = ( double ) _l1_norm;
            l2_norm = ( double ) _l2_norm;
            max_norm = ( double ) _max_norm;
            h = ( double ) _h;
            objects_match = true;
         }
         if( firstObjectType == "tnlGrid< 2, tnlFloat, tnlHost, int >" ||
             firstObjectType == "tnlGrid< 2, tnlFloat, tnlCuda, int >" )
         {
            tnlFloat _l1_norm, _l2_norm, _max_norm;
            tnlFloat _space_step = ( tnlFloat ) space_step;
            tnlFloat _h = ( tnlFloat ) h;
            compareGrids< 2, tnlFloat, tnlFloat, tnlHost, tnlHost, int >( firstFileName,
                                                                          secondFileName,
                                                                          write_difference,
                                                                          edge_skip,
                                                                          _space_step,
                                                                          _l1_norm,
                                                                          _l2_norm,
                                                                          _max_norm,
                                                                          _h,
                                                                          verbose );
            l1_norm = ( double ) _l1_norm;
            l2_norm = ( double ) _l2_norm;
            max_norm = ( double ) _max_norm;
            h = ( double ) _h;
            objects_match = true;
         }

         if( firstObjectType == "tnlGrid< 3, double, tnlHost, int >" ||
             firstObjectType == "tnlGrid< 3, double, tnlCuda, int >" )
         {
            dbgCout( "Processing file with tnlGrid3D< double > ..." );
            dbgExpr( firstFileName );
            dbgExpr( secondFileName );
            tnlGrid< 3, double, tnlHost, int > u1( "u1" ), u2( "u2" ), difference( "difference" );
            if( write_difference &&
                ! difference. setLike( u1 ) )
            {
               cerr << "I do not have enough memory to allocate the differencing grid." << endl;
               return EXIT_FAILURE;
            }
            if( ! compareObjects( u1,
                                  u2,
                                  l1_norm,
                                  l2_norm,
                                  max_norm,
                                  difference,
                                  edge_skip) )
            {
               continue;
            }
            if( space_step ) h = space_step;
            else h = Min( u1. getSpaceSteps(). x(), 
                          Min( u1. getSpaceSteps(). y(),
                               u1. getSpaceSteps(). z() ) );
            objects_match = true;
         }
         if( ! objects_match )
         {
            cerr << "Files " << firstFileName << " and " << secondFileName << " cannot be compared ... SKIPPING" << endl;
            continue;
         }

         //cout << "\r" << flush;
         l1_int += tau * l1_norm;
         l2_int += tau * l2_norm * l2_norm;
         max_int = Max( max_int, max_norm );
         if( verbose )
            WriteLine( cout, i, tau, h, l1_norm, l2_norm, max_norm );
         if( output_file )
            WriteLine( output_file, i, tau, h, l1_norm, l2_norm, max_norm );
         if( write_graph )
            WriteGraphLine( graph_file, i, tau, l1_norm, l2_norm, max_norm );
         if( write_log_graph )
            WriteLogGraphLine( log_graph_file, i, tau, l1_norm, l2_norm, max_norm );

      }
      l2_int = sqrt( l2_int );
      if( tau && verbose )
         WriteLastLine( cout, h, l1_int, l2_int, max_int );
      if( output_file )
      {
         if( tau )
            WriteLastLine( output_file, h, l1_int, l2_int, max_int );
         output_file. close();
      }
      if( verbose )
         cout << endl << "Bye." << endl;
      return 0;
   }
   if( test == "mean-curvature-circle" ||
       test == "preserved-circle" ||
       test == "willmore-circle" )
   {
      double init_radius = parameters. GetParameter< double >( "initial-radius" );
      double t = 0.0;
      for( int i = 0; i < size; i ++ )
      {
         //cout << setw( 85 ) << " ";
         const char* first_file = first_files[ i ]. getString();
         cout << "Processing file " << first_file << " ...          \r" << flush << endl;
         tnlCurve< tnlVector< 2, double > > curve( "tnl-err-norms:curve" );
         tnlFile file;
         if( ! file. open( first_files[ i ], tnlReadMode ) ||
             ! curve. load( file ) )
            return 1;
         file. close();
         if( space_step ) h = space_step;
         else h = 1.0 / ( double ) curve. getSize();
         double exact_r = init_radius;
         if( t > 0.0 )
         {
            if( test == "mean-curvature-circle" )
               exact_r = sqrt( init_radius * init_radius - 2.0 * t );
            if( test == "willmore-circle" )
               exact_r = pow( pow( init_radius, 4.0 ) + 2.0 * t, 0.25 );
         }
        
         cout << "Exact radius is " << exact_r << endl; 
         int j;
         double l1_norm( 0.0 ), l2_norm( 0.0 ), max_norm( 0.0 );
         if( curve. getSize() == 1 )
         {
            cout << "The curve is too short." << endl;
            continue;
         }
         if( write_exact_curve )
         {
            tnlString file_name = tnlString( first_file ) + tnlString( ".exact.crv.gplt" );
            cout << "Writing file " << file_name << endl;
            fstream file;
            file. open( file_name. getString(), ios :: out );
            const int curve_nodes = curve. getSize();
            const double dt = 2.0 * M_PI / ( double ) curve_nodes;
            double t( 0.0 );
            for( t = 0.0; t <= 2.0 * M_PI; t += dt )
               file << exact_r * sin( t ) << " " << exact_r * cos( t ) << endl;
            file. close();
         }

         tnlGrid< 1, double > difference( "difference" );
         difference. setDimensions( tnlVector< 1, int >( curve. getSize() ) );
         difference. setDomain( tnlVector< 1, double >( 0.0 ), 
                                tnlVector< 1, double >( 1.0 ) );
         for( j = 0; j < curve. getSize() - 1; j ++ )
         {
            if( curve[ j ]. separator ) continue;
            tnlVector< 2, double > v1 = curve[ j ]. position;
            tnlVector< 2, double > v2 = curve[ j + 1 ]. position;
            double approx_r = sqrt( v1[ 0 ] * v1[ 0 ] + v1[ 1 ] * v1[ 1 ] );
            double err = fabs( approx_r - exact_r );
            difference. setElement( j, err );
            //cout << err << endl;
            tnlVector< 2, double > dv;
            dv[ 0 ] = v1[ 0 ] - v2[ 0 ];
            dv[ 1 ] = v1[ 1 ] - v2[ 1 ];
            double h = sqrt( dv[ 0 ] * dv[ 0 ] + dv[ 1 ] * dv[ 1 ] );
            
            l1_norm += err * h;
            l2_norm += err * err * h;
            max_norm = Max( max_norm, err );

         }
         if( write_difference )
         {
            tnlString file_name( first_file );
            int strln = strlen( first_file );
            if( strcmp( file_name. getString() + strln - 3, ".gz" ) == 0 )
               file_name. setString( first_file, 0, 3 );
            else
               if( strcmp( file_name. getString() + strln - 4, ".bz2" ) == 0 )
                  file_name. setString( first_file, 0, 4 );
            file_name += ".diff";   
            tnlString file_format;
            if( strcmp( first_file + strln - 3, ".gz" ) == 0 )
               file_format. setString( "bin-gz" );
            else
               if( strcmp( first_file + strln - 4, ".bz2" ) == 0 )
               file_format. setString( "bin-bz2" );  
            file_format. setString( "gnuplot" );
            cout << "Writing file " << file_name << " ...          \r" << flush;
            difference. draw( file_name, file_format );

         }
         l2_norm = sqrt( l2_norm );
         l1_int += tau * l1_norm;
         l2_int += tau * l2_norm * l2_norm;
         max_int = Max( max_int, max_norm );
         if( verbose )
            WriteLine( cout, i, tau, h, l1_norm, l2_norm, max_norm );
         if( output_file )
            WriteLine( output_file, i, tau, h, l1_norm, l2_norm, max_norm );
         if( write_graph )
            WriteGraphLine( graph_file, i, tau, l1_norm, l2_norm, max_norm );
         if( write_log_graph )
            WriteLogGraphLine( log_graph_file, i, tau, l1_norm, l2_norm, max_norm );
         t += tau;
      }
      l2_int = sqrt( l2_int );
      if( tau && verbose )
         WriteLastLine( cout, h, l1_int, l2_int, max_int );
      if( output_file )
      {
         if( tau )
            WriteLastLine( output_file, h, l1_int, l2_int, max_int );
         output_file. close();
      }
      if( verbose )
         cout << endl << "Bye." << endl;
      if( output_file ) output_file. close();
      if( graph_file ) graph_file. close();
      if( log_graph_file ) log_graph_file. close();
      return 0;
   }
   cerr << endl << "Uknown test " << test << ". Bye." << endl;
   return -1;
}
