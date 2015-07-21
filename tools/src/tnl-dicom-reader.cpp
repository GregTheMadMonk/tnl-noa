/***************************************************************************
                          tnl-dicom-reader.cpp  -  description
                             -------------------
    begin                : Jul 19, 2015
    copyright            : (C) 2015 by Tomas Oberhuber
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

#include <config/tnlConfigDescription.h>
#include <config/tnlParameterContainer.h>

void setupConfig( tnlConfigDescription& config )
{
/*   config.addEntry< tnlString >( "mesh", "Input mesh file.", "mesh.tnl" );
   config.addRequiredEntry< tnlList< tnlString > >( "input-files", "The first set of the input files." );
   config.addEntry< tnlString >( "output-file", "File for the output data.", "tnl-diff.log" );
   config.addEntry< tnlString >( "mode", "Mode 'couples' compares two subsequent files. Mode 'sequence' compares the input files against the first one. 'halves' compares the files from the and the second half of the intput files.", "couples" );
      config.addEntryEnum< tnlString >( "couples" );
      config.addEntryEnum< tnlString >( "sequence" );
      config.addEntryEnum< tnlString >( "halves" );
   config.addEntry< bool >( "write-difference", "Write difference grid function.", false );
   config.addEntry< bool >( "write-exact-curve", "Write exact curve with given radius.", false );
   config.addEntry< int >( "edges-skip", "Width of the edges that will be skipped - not included into the error norms.", 0 );
   config.addEntry< bool >( "write-graph", "Draws a graph in the Gnuplot format of the dependence of the error norm on t.", true );
   config.addEntry< bool >( "write-log-graph", "Draws a logarithmic graph in the Gnuplot format of the dependence of the error norm on t.", true );
   config.addEntry< double >( "snapshot-period", "The period between consecutive snapshots.", 0.0 );
   config.addEntry< int >( "verbose", "Sets verbosity.", 1 );
 */
}

int main( int argc, char* argv[] )
{
   tnlParameterContainer parameters;
   tnlConfigDescription conf_desc;
   setupConfig( conf_desc );
   if( ! parseCommandLine( argc, argv, conf_desc, parameters ) )
   {
      conf_desc.printUsage( argv[ 0 ] );
      return 1;
   }
}