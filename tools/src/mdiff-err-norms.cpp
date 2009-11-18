/***************************************************************************
                          diff-norm.cpp  -  description
                             -------------------
    begin                : 2007/07/05
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

#include <fstream>
#include <stdio.h>
#include <math.h>

#include <debug/tnlDebug.h>
#include <diff/mGrid1D.h>
#include <diff/mGrid2D.h>
#include <diff/drawGrid1D.h>
#include <diff/drawGrid2D.h>
#include <core/tnlConfigDescription.h>
#include <core/mParameterContainer.h>
#include <core/mCurve.h>

#include "../../mdiff-err-norms-def.h"
#include "read-file.h"
#include "compare-objects.h"

//--------------------------------------------------------------------------
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
//--------------------------------------------------------------------------
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
//--------------------------------------------------------------------------
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
//--------------------------------------------------------------------------
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
//--------------------------------------------------------------------------
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
//--------------------------------------------------------------------------
bool TryUncompressFile( const char* file_name, 
                        tnlString& uncompressed_file_name )
{
   int strln = strlen( file_name );
   uncompressed_file_name. SetString( file_name );
   if( strcmp( file_name + strln - 3, ".gz" ) == 0 )
      if( ! UnCompressFile( file_name, "gz" ) )
      {
         cerr << "Unable to uncompress the file " << file_name << "." << endl;
         return false;
      }
      else uncompressed_file_name. SetString( file_name, 0, 3 );
   if( strcmp( file_name + strln - 4, ".bz2" ) == 0 )
      if( ! UnCompressFile( file_name, "bz2" ) )
      {
         cerr << "Unable to uncompress the file " << file_name << "." << endl;
         return false;
      }
      else uncompressed_file_name. SetString( file_name, 0, 4 );
   return true;
}
//--------------------------------------------------------------------------
bool TryCompressFile( const char* file_name, 
                      const tnlString& uncompressed_file_name )
{
   int strln = strlen( file_name );
   if( strcmp( file_name + strln - 3, ".gz" ) == 0 &&
       ! CompressFile( uncompressed_file_name. Data(), "gz" ) )
   {
      cerr << "Unable to compress back the file " << file_name << "." << endl;
      return false;
   }
   if( strcmp( file_name + strln - 4, ".bz2" ) == 0 &&
       ! CompressFile( uncompressed_file_name. Data(), "bz2" ) )
   {
      cerr << "Unable to compress back the file " << file_name << "." << endl;
      return false;
   }
   return true;
}
//--------------------------------------------------------------------------
int main( int argc, char* argv[] )
{
   dbgFunctionName( "", "main" );
   dbgInit( "mdiff-err-norms.dbg" );
   mParameterContainer parameters;
   tnlConfigDescription conf_desc;

   dbgCout( "Parsing config description file ... " );
   if( conf_desc. ParseConfigDescription( CONFIG_DESCRIPTION_FILE ) != 0 )
      return 1;
   if( ! ParseCommandLine( argc, argv, conf_desc, parameters ) )
   {
      conf_desc. PrintUsage( argv[ 0 ] );
      return 1;
   }

   cout << "************************************************************************" << endl;
   cout << "*                                                                      *" << endl;
   cout << "*           mDiff Tools: grid difference                               *" << endl;
   cout << "*                                                                      *" << endl;
   cout << "************************************************************************" << endl;

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
   long int size = first_files. Size();
   cout << "Processing " << size << " files. " << endl;

   long int edge_skip = parameters. GetParameter< int >( "edges-skip" );
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
      output_file. open( output_file_name. Data(), ios :: out );
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
   
   
   long int i;
   mGrid2D< double > u1, u2;
   double l1_int( 0.0 ), l2_int( 0.0 ), max_int( 0.0 ), h( 0.0 );
   if( test == "function" )
   {
      if( size / first2second_ratio != second_files. Size() && second_files. Size() != 1 )
      {
         cerr << "Sorry, the number of the files in the second set does not agree with the number of the files in the second one!" << endl;
         return 1;
      }
      const char* second_file = second_files[ 0 ]. Data();
      tnlString first_file_uncompressed, first_object_type,
              second_file_uncompressed, second_object_type;
      for( i = 0; i < size; i ++ )
      {
         //cout << setw( 85 ) << " ";
         const char* first_file = first_files[ i ]. Data();
         cout << "Processing file " << first_file << " ...          \r" << flush;
         if( ! TryUncompressFile( first_file, first_file_uncompressed ) )
         {
            cerr << "SKIPPING ... " << endl;
            continue;
         }
         tnlString first_object_type;
         if( ! GetObjectType( first_file_uncompressed. Data(), first_object_type ) )
         {
            cerr << "unknown object ... SKIPPING!" << endl;
            continue;
         }
         if( i % first2second_ratio == 0 && i / first2second_ratio < second_files. Size() )
         {
            second_file = second_files[ i / first2second_ratio ]. Data();
            cout << "Processing file " << second_file << " ...          \r" << flush;
            if( ! TryUncompressFile( second_file, second_file_uncompressed ) )
            {
               cerr << "SKIPPING ... " << endl;
               continue;
            }
            if( ! GetObjectType( second_file_uncompressed. Data(), second_object_type ) )
            {
               cerr << "unknown object ... SKIPPING!" << endl;
               continue;
            }
         }
         if( first_object_type != second_object_type )
         {
            cerr << "Files " << first_file << " and " << second_file << " contain different objects ... SKIPPING" << endl;
            continue;
         }
         
         double l1_norm( 0.0 ), l2_norm( 0.0 ), max_norm( 0.0 ), h( 0.0 );
         
         if( first_object_type == "mGrid2D< double >" )
         {
            dbgCout( "Processing file with mGrid2D< double > ..." );
            dbgExpr( first_file_uncompressed );
            dbgExpr( second_file_uncompressed );
            mGrid2D< double > u1, u2, difference;
            fstream file;
            file. open( first_file_uncompressed. Data(), ios :: in | ios :: binary );
            if( ! u1. Load( file ) )
            {
               cout << " unable to restore the data " << endl;
               file. close();
               return false;
            }
            file. close();
            file. open( second_file_uncompressed. Data(), ios :: in | ios :: binary );
            if( ! u2. Load( file ) )
            {
               cout << " unable to restore the data " << endl;
               file. close();
               return false;
            }
            file. close();
            if( write_difference )
            {
               difference. SetNewDimensions( u1 );
               difference. SetNewDomain( u1 );
            }
            if( ! Compare( u1,
                           u2,
                           l1_norm,
                           l2_norm,
                           max_norm,
                           difference ) )
            {
               continue;
            }
            if( space_step ) h = space_step;
            else h = Min( u1. GetHx(), u1. GetHy() );
         }
         if( first_object_type == "mGrid3D< double >" )
         {
            dbgCout( "Processing file with mGrid3D< double > ..." );
            dbgExpr( first_file_uncompressed );
            dbgExpr( second_file_uncompressed );
            mGrid3D< double > u1, u2, difference;
            fstream file;
            file. open( first_file_uncompressed. Data(), ios :: in | ios :: binary );
            if( ! u1. Load( file ) )
            {
               cout << " unable to restore the data " << endl;
               file. close();
               return false;
            }
            file. close();
            file. open( second_file_uncompressed. Data(), ios :: in | ios :: binary );
            if( ! u2. Load( file ) )
            {
               cout << " unable to restore the data " << endl;
               file. close();
               return false;
            }
            file. close();
            if( write_difference )
            {
               difference. SetNewDimensions( u1 );
               difference. SetNewDomain( u1 );
            }
            if( ! Compare( u1,
                           u2,
                           l1_norm,
                           l2_norm,
                           max_norm,
                           difference ) )
            {
               continue;
            }
            if( space_step ) h = space_step;
            else h = Min( u1. GetHx(), u1. GetHy() );
         }
         //cout << "\r" << flush;
         l1_int += tau * l1_norm;
         l2_int += tau * l2_norm * l2_norm;
         max_int = Max( max_int, max_norm );
         WriteLine( cout, i, tau, h, l1_norm, l2_norm, max_norm );
         if( output_file )
            WriteLine( output_file, i, tau, h, l1_norm, l2_norm, max_norm );
         if( write_graph )
            WriteGraphLine( graph_file, i, tau, l1_norm, l2_norm, max_norm );
         if( write_log_graph )
            WriteLogGraphLine( log_graph_file, i, tau, l1_norm, l2_norm, max_norm );

         cout << "Compressing opened files back ...                   \r" << flush;
         TryCompressFile( first_file, first_file_uncompressed );
         TryCompressFile( second_file, second_file_uncompressed );

      }
      l2_int = sqrt( l2_int );
      if( tau )
         WriteLastLine( cout, h, l1_int, l2_int, max_int );
      if( output_file )
      {
         if( tau )
            WriteLastLine( output_file, h, l1_int, l2_int, max_int );
         output_file. close();
      }
      cout << endl << "Bye." << endl;
      return 0;
   }
   if( test == "mean-curvature-circle" ||
       test == "preserved-circle" ||
       test == "willmore-circle" )
   {
      double init_radius = parameters. GetParameter< double >( "initial-radius" );
      double t = 0.0;
      for( i = 0; i < size; i ++ )
      {
         //cout << setw( 85 ) << " ";
         const char* first_file = first_files[ i ]. Data();
         cout << "Processing file " << first_file << " ...          \r" << flush << endl;
         mCurve< mVector< 2, double > > curve;
         if( ! ReadFile( first_file, curve ) ) return 1;
         if( space_step ) h = space_step;
         else h = 1.0 / ( double ) curve. Size();
         double exact_r = init_radius;
         if( t > 0.0 )
         {
            if( test == "mean-curvature-circle" )
               exact_r = sqrt( init_radius * init_radius - 2.0 * t );
            if( test == "willmore-circle" )
               exact_r = pow( pow( init_radius, 4.0 ) + 2.0 * t, 0.25 );
         }
        
         cout << "Exact radius is " << exact_r << endl; 
         long int j;
         double l1_norm( 0.0 ), l2_norm( 0.0 ), max_norm( 0.0 );
         if( curve. Size() == 1 )
         {
            cout << "The curve is too short." << endl;
            continue;
         }
         if( write_exact_curve )
         {
            tnlString file_name = tnlString( first_file ) + tnlString( ".exact.crv.gplt" );
            cout << "Writing file " << file_name << endl;
            fstream file;
            file. open( file_name. Data(), ios :: out );
            const long int curve_nodes = curve. Size();
            const double dt = 2.0 * M_PI / ( double ) curve_nodes;
            double t( 0.0 );
            for( t = 0.0; t <= 2.0 * M_PI; t += dt )
               file << exact_r * sin( t ) << " " << exact_r * cos( t ) << endl;
            file. close();
         }

         mGrid1D< double > difference( curve. Size(), 0.0, 1.0 );
         for( j = 0; j < curve. Size() - 1; j ++ )
         {
            if( curve[ j ]. separator ) continue;
            mVector< 2, double > v1 = curve[ j ]. position;
            mVector< 2, double > v2 = curve[ j + 1 ]. position;
            double approx_r = sqrt( v1[ 0 ] * v1[ 0 ] + v1[ 1 ] * v1[ 1 ] );
            double err = fabs( approx_r - exact_r );
            difference( j ) = err;
            //cout << err << endl;
            mVector< 2, double > dv;
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
            if( strcmp( file_name. Data() + strln - 3, ".gz" ) == 0 )
               file_name. SetString( first_file, 0, 3 );
            else
               if( strcmp( file_name. Data() + strln - 4, ".bz2" ) == 0 )
                  file_name. SetString( first_file, 0, 4 );
            file_name += ".diff";   
            tnlString file_format;
            if( strcmp( first_file + strln - 3, ".gz" ) == 0 )
               file_format. SetString( "bin-gz" );
            else
               if( strcmp( first_file + strln - 4, ".bz2" ) == 0 )
               file_format. SetString( "bin-bz2" );  
            file_format. SetString( "gnuplot" );
            cout << "Writing file " << file_name << " ...          \r" << flush;
            Draw( difference, file_name. Data(), file_format. Data() );

         }
         l2_norm = sqrt( l2_norm );
         l1_int += tau * l1_norm;
         l2_int += tau * l2_norm * l2_norm;
         max_int = Max( max_int, max_norm );
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
      if( tau )
         WriteLastLine( cout, h, l1_int, l2_int, max_int );
      if( output_file )
      {
         if( tau )
            WriteLastLine( output_file, h, l1_int, l2_int, max_int );
         output_file. close();
      }
      cout << endl << "Bye." << endl;
      if( output_file ) output_file. close();
      if( graph_file ) graph_file. close();
      if( log_graph_file ) log_graph_file. close();
      return 0;
   }
   cout << endl << "Uknown test " << test << ". Bye." << endl;
   return -1;
}

