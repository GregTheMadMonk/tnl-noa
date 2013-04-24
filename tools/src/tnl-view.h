/***************************************************************************
                          tnl-view.h  -  description
                             -------------------
    begin                : Jan 21, 2013
    copyright            : (C) 2013 by Tomas Oberhuber
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

#ifndef TNL_VIEW_H_
#define TNL_VIEW_H_

#include <cstdlib>
#include <core/mfilename.h>
#include <config/tnlParameterContainer.h>
#include <core/tnlString.h>
#include <core/tnlVector.h>
#include <core/tnlMultiVector.h>
#include <mesh/tnlGrid.h>

using namespace std;

template< typename Mesh, typename Element, typename Index, int Dimensions >
bool convertObject( const Mesh& mesh,
                    const tnlString& inputFileName,
                    const tnlList< tnlString >& parsedObjectType,
                    const tnlParameterContainer& parameters )
{
   int verbose = parameters. GetParameter< int >( "verbose");
   tnlString outputFileName( inputFileName );
   RemoveFileExtension( outputFileName );
   tnlString outputFormat = parameters. GetParameter< tnlString >( "output-format" );
   if( outputFormat == "gnuplot" )
      outputFileName += ".gplt";
   else
   {
      cerr << "Unknown file format " << outputFormat << ".";
      return false;
   }
   if( verbose )
      cout << " writing to " << outputFileName << " ... " << flush;


   if( parsedObjectType[ 0 ] == "tnlSharedVector" ||
       parsedObjectType[ 0 ] == "tnlVector" )
   {
      tnlVector< Element, tnlHost, Index > vector;
      if( ! vector. load( inputFileName ) )
         return false;
      if( ! mesh. write( vector, outputFileName, outputFormat ) )
         return false;
   }

   if( parsedObjectType[ 0 ] == "tnlMultiVector" ||
       parsedObjectType[ 0 ] == "tnlSharedMultiVector" )
   {
      tnlMultiVector< Dimensions, Element, tnlHost, Index > multiVector;
      if( ! multiVector. load( inputFileName ) )
         return false;
      tnlGrid< Dimensions, Element, tnlHost, Index > grid;
      grid. setDimensions( multiVector. getDimensions() );
      grid. setLowerCorner( tnlTuple< Dimensions, Element >( 0.0 ) );
      grid. setUpperCorner( tnlTuple< Dimensions, Element >( 1.0 ) );
      const Element spaceStep = grid. getSpaceStep(). x();
      grid. setSpaceStep( tnlTuple< Dimensions, Element >( spaceStep ) );
      if( ! grid. write( multiVector, outputFileName, outputFormat ) )
         return false;
   }
   if( verbose )
      cout << "[ OK ].  \r";
   return true;
}

template< typename Mesh, typename Element, typename Index >
bool setDimensions( const Mesh& mesh,
                    const tnlString& inputFileName,
                    const tnlList< tnlString >& parsedObjectType,
                    const tnlParameterContainer& parameters )
{
   int dimensions( 0 );
   if( parsedObjectType[ 0 ] == "tnlMultiVector" ||
       parsedObjectType[ 0 ] == "tnlSharedMultiVector" )
      dimensions = atoi( parsedObjectType[ 1 ]. getString() );
   if( parsedObjectType[ 0 ] == "tnlVector" ||
       parsedObjectType[ 0 ] == "tnlSharedVector" )
      dimensions = 1;
   switch( dimensions )
   {
      case 1:
         return convertObject< Mesh, Element, Index, 1 >( mesh, inputFileName, parsedObjectType, parameters );
      case 2:
         return convertObject< Mesh, Element, Index, 2 >( mesh, inputFileName, parsedObjectType, parameters );
      case 3:
         return convertObject< Mesh, Element, Index, 3 >( mesh, inputFileName, parsedObjectType, parameters );
   }
   cerr << "Cannot convert objects with " << dimensions << " dimensions." << endl;
   return false;
}

template< typename Mesh, typename Element >
bool setIndexType( const Mesh& mesh,
                   const tnlString& inputFileName,
                   const tnlList< tnlString >& parsedObjectType,
                   const tnlParameterContainer& parameters )
{
   tnlString indexType;
   if( parsedObjectType[ 0 ] == "tnlMultiVector" ||
       parsedObjectType[ 0 ] == "tnlSharedMultiVector" )
      indexType = parsedObjectType[ 4 ];
   if( parsedObjectType[ 0 ] == "tnlSharedVector" ||
       parsedObjectType[ 0 ] == "tnlVector" )
      indexType = parsedObjectType[ 3 ];

   if( indexType == "int" )
      return setDimensions< Mesh, Element, int >( mesh, inputFileName, parsedObjectType, parameters );
   if( indexType == "long-int" )
      return setDimensions< Mesh, Element, long int >( mesh, inputFileName, parsedObjectType, parameters );
   cerr << "Unknown index type " << indexType << "." << endl;
   return false;
}

template< typename Mesh >
bool setElementType( const Mesh& mesh,
                     const tnlString& inputFileName,
                     const tnlList< tnlString >& parsedObjectType,
                     const tnlParameterContainer& parameters )
{
   tnlString elementType;
   if( parsedObjectType[ 0 ] == "tnlMultiVector" ||
       parsedObjectType[ 0 ] == "tnlSharedMultiVector" )
      elementType = parsedObjectType[ 2 ];
   if( parsedObjectType[ 0 ] == "tnlSharedVector" ||
       parsedObjectType[ 0 ] == "tnlVector" )
      elementType = parsedObjectType[ 1 ];

   if( elementType == "float" )
      return setIndexType< Mesh, float >( mesh, inputFileName, parsedObjectType, parameters );
   if( elementType == "double" )
      return setIndexType< Mesh, double >( mesh, inputFileName, parsedObjectType, parameters );
   if( elementType == "long-double" )
      return setIndexType< Mesh, long double >( mesh, inputFileName, parsedObjectType, parameters );
   cerr << "Unknown element type " << elementType << "." << endl;
   return false;
}

template< typename Mesh >
bool processFiles( const tnlParameterContainer& parameters )
{
   int verbose = parameters. GetParameter< int >( "verbose");
   tnlString meshFile = parameters. GetParameter< tnlString >( "mesh" );
   Mesh mesh;
   if( meshFile != "" )
      if( ! mesh. load( meshFile ) )
      {
         cerr << "I am not able to load mesh from the file " << meshFile << "." << endl;
         return false;
      }

   tnlList< tnlString > inputFiles = parameters. GetParameter< tnlList< tnlString > >( "input-files" );
   for( int i = 0; i < inputFiles. getSize(); i ++ )
   {
      if( verbose )
         cout << "Processing file " << inputFiles[ i ] << " ... " << flush;

      tnlString objectType;
      if( ! getObjectType( inputFiles[ i ], objectType ) )
          cerr << "unknown object ... SKIPPING!" << endl;
      else
      {
         if( verbose )
            cout << objectType << " detected ... ";

         tnlList< tnlString > parsedObjectType;
         if( ! parseObjectType( objectType, parsedObjectType ) )
         {
            cerr << "Unable to parse object type " << objectType << "." << endl;
            return false;
         }
         setElementType< Mesh >( mesh, inputFiles[ i ], parsedObjectType, parameters );
      }
   }
   if( verbose )
      cout << endl;
}


#endif /* TNL_VIEW_H_ */
