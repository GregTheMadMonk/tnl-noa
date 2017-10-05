/***************************************************************************
                          tnlCurve2gnuplot.cpp  -  description
                             -------------------
    begin                : 2007/12/16
    copyright            : (C) 2007 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#include <TNL/Config/ParameterContainer.h>
#include <TNL/Curve.h>
#include <TNL/Containers/StaticVector.h>

using namespace TNL;

void setupConfig( Config::ConfigDescription& config )
{
   config.addDelimiter                            ( "General settings:" );
   config.addRequiredList< String >(  "input-files", "Input files." );
   config.addList< String >( "output-files", "Output files." );
   config.addEntry< int >( "output-step", "Decrease number of the output curve nodes." );
   config.addEntry< String >( "output-file-format", "Output file format. Can be gnuplot.", "gnuplot" );
}

//--------------------------------------------------------------------------
int main( int argc, char* argv[] )
{
   Config::ParameterContainer parameters;
   Config::ConfigDescription conf_desc;
 
   setupConfig( conf_desc );
   if( ! parseCommandLine( argc, argv, conf_desc, parameters ) )
   {
      conf_desc.printUsage( argv[ 0 ] );
      return 1;
   }

   Containers::List< String > input_files = parameters. getParameter< Containers::List< String > >( "input-files" );
   Containers::List< String > output_files;
   if( ! parameters. getParameter< Containers::List< String > >( "output-files", output_files ) )
      std::cout << "No output files were given." << std::endl;
   int output_step( 1 );
   parameters. getParameter< int >( "output-step", output_step );
   String output_file_format = parameters. getParameter< String >( "output-file-format" );

   int size = input_files. getSize();
   /*if( size != output_files. getSize() )
   {
      std::cerr << "Sorry, there is different number of input and output files." << std::endl;
      return 1;
   }*/
   int i;
   Curve< Containers::StaticVector< 2, double > > crv( "tnlcurve2gnuplot:curve" );
   for( i = 0; i < size; i ++ )
   {
      const char* input_file = input_files[ i ]. getString();
      std::cout << "Processing file " << input_file << " ... " << std::flush;
 
      File file;
      if( ! file. open( input_files[ i ], IOMode::read ) )
      {
         std::cout << " unable to open file " << input_files[ i ] << std::endl;
         continue;
      }
      if( ! crv. load( file ) )
      {
         std::cout << " unable to restore the data " << std::endl;
         continue;
      }
      file. close();

      Curve< Containers::StaticVector< 2, double > > out_crv( "tnlcurve2gnuplot:outcurve" );
      const int size = crv. getSize();
      int i;
      for( i = 0; i < size; i += output_step )
      {
         out_crv. Append( crv[ i ]. position, crv[ i ]. separator );
         //StaticVector< 2, double > v = crv[ i ]. position;
         //v[ 0 ] = u( i );
         //v[ 1 ] = u( i + 1 );
         //out_crv. Append( v );
      }

      String output_file_name;
      if( ! output_files. isEmpty() ) output_file_name = output_files[ i ];
      else
      {
         if( output_file_format == "gnuplot" )
            output_file_name += ".gplt";
      }
      std::cout << " writing... " << output_file_name << std::endl;
      if( ! Write( out_crv, output_file_name. getString(), output_file_format. getString() ) )
      {
         std::cerr << " unable to write to " << output_file_name << std::endl;
      }
   }
}
