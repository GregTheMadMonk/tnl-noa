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

bool getOutputFileName( const tnlString& inputFileName,
                        const tnlString& outputFormat,
                        tnlString& outputFileName )
{
   outputFileName = inputFileName;
   RemoveFileExtension( outputFileName );
   if( outputFormat == "gnuplot" )
   {
      outputFileName += ".gplt";
      return true;
   }
   else
   {
      cerr << "Unknown file format " << outputFormat << ".";
      return false;
   }
}

template< typename Mesh, typename Element, typename Real, typename Index, int Dimensions >
bool convertObject( const Mesh& mesh,
                    const tnlString& inputFileName,
                    const tnlList< tnlString >& parsedObjectType,
                    const tnlParameterContainer& parameters )
{
   int verbose = parameters. GetParameter< int >( "verbose");
   tnlString outputFormat = parameters. GetParameter< tnlString >( "output-format" );
   tnlString outputFileName;
   if( ! getOutputFileName( inputFileName,
                            outputFormat,
                            outputFileName ) )
      return false;
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
      tnlGrid< Dimensions, Real, tnlHost, Index > grid;
      grid. setDimensions( multiVector. getDimensions() );
      grid. setOrigin( tnlTuple< Dimensions, Real >( 0.0 ) );
      grid. setProportions( tnlTuple< Dimensions, Real >( 1.0 ) );
      const Real spaceStep = grid. getParametricStep(). x();
      grid. setParametricStep( tnlTuple< Dimensions, Real >( spaceStep ) );
      if( ! grid. write( multiVector, outputFileName, outputFormat ) )
         return false;
   }
   if( verbose )
      cout << "[ OK ].  \r";
   return true;
}

template< typename Mesh, typename Element, typename Real, typename Index >
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
         return convertObject< Mesh, Element, Real, Index, 1 >( mesh, inputFileName, parsedObjectType, parameters );
      case 2:
         return convertObject< Mesh, Element, Real, Index, 2 >( mesh, inputFileName, parsedObjectType, parameters );
      case 3:
         return convertObject< Mesh, Element, Real, Index, 3 >( mesh, inputFileName, parsedObjectType, parameters );
   }
   cerr << "Cannot convert objects with " << dimensions << " dimensions." << endl;
   return false;
}

template< typename Mesh, typename Element, typename Real >
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
      return setDimensions< Mesh, Element, Real, int >( mesh, inputFileName, parsedObjectType, parameters );
   if( indexType == "long-int" )
      return setDimensions< Mesh, Element, Real, long int >( mesh, inputFileName, parsedObjectType, parameters );
   cerr << "Unknown index type " << indexType << "." << endl;
   return false;
}

template< typename Mesh >
bool setTupleType( const Mesh& mesh,
                   const tnlString& inputFileName,
                   const tnlList< tnlString >& parsedObjectType,
                   const tnlList< tnlString >& parsedElementType,
                   const tnlParameterContainer& parameters )
{
   int dimensions = atoi( parsedElementType[ 1 ].getString() );
   tnlString dataType = parsedElementType[ 2 ];
   if( dataType == "float" )
      switch( dimensions )
      {
         case 1:
            return setIndexType< Mesh, tnlTuple< 1, float >, float >( mesh, inputFileName, parsedObjectType, parameters );
            break;
         case 2:
            return setIndexType< Mesh, tnlTuple< 2, float >, float >( mesh, inputFileName, parsedObjectType, parameters );
            break;
         case 3:
            return setIndexType< Mesh, tnlTuple< 3, float >, float >( mesh, inputFileName, parsedObjectType, parameters );
            break;
      }
   if( dataType == "double" )
      switch( dimensions )
      {
         case 1:
            return setIndexType< Mesh, tnlTuple< 1, double >, double >( mesh, inputFileName, parsedObjectType, parameters );
            break;
         case 2:
            return setIndexType< Mesh, tnlTuple< 2, double >, double >( mesh, inputFileName, parsedObjectType, parameters );
            break;
         case 3:
            return setIndexType< Mesh, tnlTuple< 3, double >, double >( mesh, inputFileName, parsedObjectType, parameters );
            break;
      }
   if( dataType == "long double" )
      switch( dimensions )
      {
         case 1:
            return setIndexType< Mesh, tnlTuple< 1, long double >, long double >( mesh, inputFileName, parsedObjectType, parameters );
            break;
         case 2:
            return setIndexType< Mesh, tnlTuple< 2, long double >, long double >( mesh, inputFileName, parsedObjectType, parameters );
            break;
         case 3:
            return setIndexType< Mesh, tnlTuple< 3, long double >, long double >( mesh, inputFileName, parsedObjectType, parameters );
            break;
      }
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
      return setIndexType< Mesh, float, float >( mesh, inputFileName, parsedObjectType, parameters );
   if( elementType == "double" )
      return setIndexType< Mesh, double, double >( mesh, inputFileName, parsedObjectType, parameters );
   if( elementType == "long double" )
      return setIndexType< Mesh, long double, long double >( mesh, inputFileName, parsedObjectType, parameters );
   tnlList< tnlString > parsedElementType;
   if( ! parseObjectType( elementType, parsedElementType ) )
   {
      cerr << "Unable to parse object type " << elementType << "." << endl;
      return false;
   }
   if( parsedElementType[ 0 ] == "tnlTuple" )
      return setTupleType< Mesh >( mesh, inputFileName, parsedObjectType, parsedElementType, parameters );

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
   mesh. writeMesh( "mesh.asy", "asymptote" );

   bool checkOutputFile = parameters. GetParameter< bool >( "check-output-file" );
   tnlList< tnlString > inputFiles = parameters. GetParameter< tnlList< tnlString > >( "input-files" );
#ifdef HAVE_OPENMP
#pragma omp parallel for
#endif
   for( int i = 0; i < inputFiles. getSize(); i ++ )
   {
      if( verbose )
         cout << "Processing file " << inputFiles[ i ] << " ... " << flush;

      tnlString outputFormat = parameters. GetParameter< tnlString >( "output-format" );
      tnlString outputFileName;
      if( ! getOutputFileName( inputFiles[ i ],
                               outputFormat,
                               outputFileName ) )
         return false;
      if( checkOutputFile && fileExists( outputFileName ) )
      {
         if( verbose )
            cout << " file already exists. Skipping.            \r" << flush;
         continue;
      }

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
