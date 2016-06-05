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
#include <core/vectors/tnlVector.h>
#include <core/vectors/tnlMultiVector.h>
#include <mesh/tnlGrid.h>
#include <functions/tnlMeshFunction.h>

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


template< typename MeshFunction >
bool writeMeshFunction( const typename MeshFunction::MeshPointer& meshPointer,
                        const tnlString& inputFileName,
                        const tnlParameterContainer& parameters  )
{
   MeshFunction function( meshPointer );
   if( ! function.load( inputFileName ) )
   {
      std::cerr << "Unable to load mesh function from a file " << inputFileName << "." << std::endl;
      return false;
   }

   int verbose = parameters. getParameter< int >( "verbose");
   tnlString outputFormat = parameters. getParameter< tnlString >( "output-format" );
   tnlString outputFileName;
   if( ! getOutputFileName( inputFileName,
                            outputFormat,
                            outputFileName ) )
      return false;
   if( verbose )
      cout << " writing to " << outputFileName << " ... " << flush;

   return function.write( outputFileName, outputFormat );
}

template< typename MeshPointer,
          int EntityDimensions,
          typename Real >
bool setMeshFunctionRealType( const MeshPointer& meshPointer,
                              const tnlString& inputFileName,
                              const tnlParameterContainer& parameters  )
{
   return writeMeshFunction< tnlMeshFunction< typename MeshPointer::ObjectType, EntityDimensions, Real > >( meshPointer, inputFileName, parameters );
}

template< typename MeshPointer,
          int EntityDimensions >
bool setMeshEntityType( const MeshPointer& meshPointer,
                        const tnlString& inputFileName,
                        const tnlList< tnlString >& parsedObjectType,
                        const tnlParameterContainer& parameters )
{
   if( parsedObjectType[ 3 ] == "float" )
      return setMeshFunctionRealType< MeshPointer, EntityDimensions, float >( meshPointer, inputFileName, parameters );
   if( parsedObjectType[ 3 ] == "double" )
      return setMeshFunctionRealType< MeshPointer, EntityDimensions, double >( meshPointer, inputFileName, parameters );
   if( parsedObjectType[ 3 ] == "long double" )
      return setMeshFunctionRealType< MeshPointer, EntityDimensions, long double >( meshPointer, inputFileName, parameters );
   std::cerr << "Unsupported arithmetics " << parsedObjectType[ 3 ] << " in mesh function " << inputFileName << std::endl;
   return false;
}

template< typename MeshReal,
          typename MeshIndex >
bool setMeshEntityDimensions( const tnlSharedPointer< tnlGrid< 1, MeshReal, tnlHost, MeshIndex > >& meshPointer,
                              const tnlString& inputFileName,
                              const tnlList< tnlString >& parsedObjectType,
                              const tnlParameterContainer& parameters )
{
   typedef tnlGrid< 1, MeshReal, tnlHost, MeshIndex > Mesh;
   typedef tnlSharedPointer< Mesh > MeshPointer;
   int meshEntityDimensions = atoi( parsedObjectType[ 2 ].getString() );
   switch( meshEntityDimensions )
   {
      case 0:
         return setMeshEntityType< MeshPointer, 0 >( meshPointer, inputFileName, parsedObjectType, parameters );
         break;      
      case 1:
         return setMeshEntityType< MeshPointer, 1 >( meshPointer, inputFileName, parsedObjectType, parameters );
         break;
      default:
         cerr << "Unsupported mesh functions entity dimensions count " << meshEntityDimensions << "." << endl;
         return false;
   }
}

template< typename MeshReal,
          typename MeshIndex >
bool setMeshEntityDimensions( const tnlSharedPointer< tnlGrid< 2, MeshReal, tnlHost, MeshIndex > >& meshPointer,
                              const tnlString& inputFileName,
                              const tnlList< tnlString >& parsedObjectType,
                              const tnlParameterContainer& parameters )
{
   typedef tnlGrid< 2, MeshReal, tnlHost, MeshIndex > Mesh;
   typedef tnlSharedPointer< Mesh > MeshPointer;
   int meshEntityDimensions = atoi( parsedObjectType[ 2 ].getString() );
   switch( meshEntityDimensions )
   {
      case 0:
         return setMeshEntityType< MeshPointer, 0 >( meshPointer, inputFileName, parsedObjectType, parameters );
         break;            
      case 1:
         return setMeshEntityType< MeshPointer, 1 >( meshPointer, inputFileName, parsedObjectType, parameters );
         break;
      case 2:
         return setMeshEntityType< MeshPointer, 2 >( meshPointer, inputFileName, parsedObjectType, parameters );
         break;
      default:
         cerr << "Unsupported mesh functions entity dimensions count " << meshEntityDimensions << "." << endl;
         return false;
   }
}

template< typename MeshReal,
          typename MeshIndex >
bool setMeshEntityDimensions( const tnlSharedPointer< tnlGrid< 3, MeshReal, tnlHost, MeshIndex > >& meshPointer,
                              const tnlString& inputFileName,
                              const tnlList< tnlString >& parsedObjectType,
                              const tnlParameterContainer& parameters )
{
   typedef tnlGrid< 3, MeshReal, tnlHost, MeshIndex > Mesh;
   typedef tnlSharedPointer< Mesh > MeshPointer;
   int meshEntityDimensions = atoi( parsedObjectType[ 2 ].getString() );
   switch( meshEntityDimensions )
   {
      case 0:
         return setMeshEntityType< MeshPointer, 0 >( meshPointer, inputFileName, parsedObjectType, parameters );
         break;      
      case 1:
         return setMeshEntityType< MeshPointer, 1 >( meshPointer, inputFileName, parsedObjectType, parameters );
         break;
      case 2:
         return setMeshEntityType< MeshPointer, 2 >( meshPointer, inputFileName, parsedObjectType, parameters );
         break;
      case 3:
         return setMeshEntityType< MeshPointer, 3 >( meshPointer, inputFileName, parsedObjectType, parameters );
         break;
      default:
         cerr << "Unsupported mesh functions entity dimensions count " << meshEntityDimensions << "." << endl;
         return false;
   }
}

template< typename MeshPointer >
bool setMeshFunction( const MeshPointer& meshPointer,
                      const tnlString& inputFileName,
                      const tnlList< tnlString >& parsedObjectType,
                      const tnlParameterContainer& parameters )
{
   if( parsedObjectType[ 1 ] != meshPointer->getSerializationType() )
   {
      cerr << "Incompatible mesh type for the mesh function " << inputFileName << "." << endl;
      return false;
   }
   return setMeshEntityDimensions( meshPointer, inputFileName, parsedObjectType, parameters );
}


template< typename MeshPointer, typename Element, typename Real, typename Index, int Dimensions >
bool convertObject( const MeshPointer& meshPointer,
                    const tnlString& inputFileName,
                    const tnlList< tnlString >& parsedObjectType,
                    const tnlParameterContainer& parameters )
{
   int verbose = parameters. getParameter< int >( "verbose");
   tnlString outputFormat = parameters. getParameter< tnlString >( "output-format" );
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
      if( ! meshPointer->write( vector, outputFileName, outputFormat ) )
         return false;
   }

   if( parsedObjectType[ 0 ] == "tnlMultiVector" ||
       parsedObjectType[ 0 ] == "tnlSharedMultiVector" )
   {
      tnlMultiVector< Dimensions, Element, tnlHost, Index > multiVector;
      if( ! multiVector. load( inputFileName ) )
         return false;
      typedef tnlGrid< Dimensions, Real, tnlHost, Index > GridType;
      typedef typename GridType::VertexType VertexType;
      typedef typename GridType::CoordinatesType CoordinatesType;
      GridType grid;
      grid. setDomain( VertexType( 0.0 ), VertexType( 1.0 ) );
      grid. setDimensions( CoordinatesType( multiVector. getDimensions() ) );
      const Real spaceStep = grid. getSpaceSteps(). x();
      if( ! grid. write( multiVector, outputFileName, outputFormat ) )
         return false;
   }
   return true;
}

template< typename MeshPointer, typename Element, typename Real, typename Index >
bool setDimensions( const MeshPointer& meshPointer,
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
         return convertObject< MeshPointer, Element, Real, Index, 1 >( meshPointer, inputFileName, parsedObjectType, parameters );
      case 2:
         return convertObject< MeshPointer, Element, Real, Index, 2 >( meshPointer, inputFileName, parsedObjectType, parameters );
      case 3:
         return convertObject< MeshPointer, Element, Real, Index, 3 >( meshPointer, inputFileName, parsedObjectType, parameters );
   }
   cerr << "Cannot convert objects with " << dimensions << " dimensions." << endl;
   return false;
}

template< typename MeshPointer, typename Element, typename Real >
bool setIndexType( const MeshPointer& meshPointer,
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
      return setDimensions< MeshPointer, Element, Real, int >( meshPointer, inputFileName, parsedObjectType, parameters );
   if( indexType == "long-int" )
      return setDimensions< MeshPointer, Element, Real, long int >( meshPointer, inputFileName, parsedObjectType, parameters );
   cerr << "Unknown index type " << indexType << "." << endl;
   return false;
}

template< typename MeshPointer >
bool setTupleType( const MeshPointer& meshPointer,
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
            return setIndexType< MeshPointer, tnlStaticVector< 1, float >, float >( meshPointer, inputFileName, parsedObjectType, parameters );
            break;
         case 2:
            return setIndexType< MeshPointer, tnlStaticVector< 2, float >, float >( meshPointer, inputFileName, parsedObjectType, parameters );
            break;
         case 3:
            return setIndexType< MeshPointer, tnlStaticVector< 3, float >, float >( meshPointer, inputFileName, parsedObjectType, parameters );
            break;
      }
   if( dataType == "double" )
      switch( dimensions )
      {
         case 1:
            return setIndexType< MeshPointer, tnlStaticVector< 1, double >, double >( meshPointer, inputFileName, parsedObjectType, parameters );
            break;
         case 2:
            return setIndexType< MeshPointer, tnlStaticVector< 2, double >, double >( meshPointer, inputFileName, parsedObjectType, parameters );
            break;
         case 3:
            return setIndexType< MeshPointer, tnlStaticVector< 3, double >, double >( meshPointer, inputFileName, parsedObjectType, parameters );
            break;
      }
   if( dataType == "long double" )
      switch( dimensions )
      {
         case 1:
            return setIndexType< MeshPointer, tnlStaticVector< 1, long double >, long double >( meshPointer, inputFileName, parsedObjectType, parameters );
            break;
         case 2:
            return setIndexType< MeshPointer, tnlStaticVector< 2, long double >, long double >( meshPointer, inputFileName, parsedObjectType, parameters );
            break;
         case 3:
            return setIndexType< MeshPointer, tnlStaticVector< 3, long double >, long double >( meshPointer, inputFileName, parsedObjectType, parameters );
            break;
      }
}

template< typename MeshPointer >
bool setElementType( const MeshPointer& meshPointer,
                     const tnlString& inputFileName,
                     const tnlList< tnlString >& parsedObjectType,
                     const tnlParameterContainer& parameters )
{
   tnlString elementType;

   // TODO: Fix this even for arrays
   if( parsedObjectType[ 0 ] == "tnlMultiVector" ||
       parsedObjectType[ 0 ] == "tnlSharedMultiVector" )
      elementType = parsedObjectType[ 2 ];
   if( parsedObjectType[ 0 ] == "tnlSharedVector" ||
       parsedObjectType[ 0 ] == "tnlVector" )
      elementType = parsedObjectType[ 1 ];


   if( elementType == "float" )
      return setIndexType< MeshPointer, float, float >( meshPointer, inputFileName, parsedObjectType, parameters );
   if( elementType == "double" )
      return setIndexType< MeshPointer, double, double >( meshPointer, inputFileName, parsedObjectType, parameters );
   if( elementType == "long double" )
      return setIndexType< MeshPointer, long double, long double >( meshPointer, inputFileName, parsedObjectType, parameters );
   tnlList< tnlString > parsedElementType;
   if( ! parseObjectType( elementType, parsedElementType ) )
   {
      cerr << "Unable to parse object type " << elementType << "." << endl;
      return false;
   }
   if( parsedElementType[ 0 ] == "tnlStaticVector" )
      return setTupleType< MeshPointer >( meshPointer, inputFileName, parsedObjectType, parsedElementType, parameters );

   cerr << "Unknown element type " << elementType << "." << endl;
   return false;
}

template< typename Mesh >
bool processFiles( const tnlParameterContainer& parameters )
{
   int verbose = parameters. getParameter< int >( "verbose");
   tnlString meshFile = parameters. getParameter< tnlString >( "mesh" );

   typedef tnlSharedPointer< Mesh > MeshPointer;
   MeshPointer meshPointer;
   meshPointer.create();
   
   if( meshFile != "" )
      if( ! meshPointer->load( meshFile ) )
      {
         cerr << "I am not able to load mesh from the file " << meshFile << "." << endl;
         return false;
      }
   meshPointer->writeMesh( "mesh.asy", "asymptote" );

   bool checkOutputFile = parameters. getParameter< bool >( "check-output-file" );
   tnlList< tnlString > inputFiles = parameters. getParameter< tnlList< tnlString > >( "input-files" );
   bool error( false );
//#ifdef HAVE_OPENMP
//#pragma omp parallel for
//#endif
   for( int i = 0; i < inputFiles. getSize(); i ++ )
   {
      if( verbose )
         cout << "Processing file " << inputFiles[ i ] << " ... " << flush;

      tnlString outputFormat = parameters. getParameter< tnlString >( "output-format" );
      tnlString outputFileName;
      if( ! getOutputFileName( inputFiles[ i ],
                               outputFormat,
                               outputFileName ) )
      {
         error = true;
         continue;
      }
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
            error = true;
            continue;
         }
         if( parsedObjectType[ 0 ] == "tnlMultiVector" ||
             parsedObjectType[ 0 ] == "tnlSharedMultiVector" ||      
             parsedObjectType[ 0 ] == "tnlSharedVector" ||
             parsedObjectType[ 0 ] == "tnlVector" )
            setElementType< MeshPointer >( meshPointer, inputFiles[ i ], parsedObjectType, parameters );
         if( parsedObjectType[ 0 ] == "tnlMeshFunction" )
            setMeshFunction< MeshPointer >( meshPointer, inputFiles[ i ], parsedObjectType, parameters );
         if( verbose )
            cout << "[ OK ].  " << endl;

      }
   }
   if( verbose )
      cout << endl;
   return ! error;
}


#endif /* TNL_VIEW_H_ */
