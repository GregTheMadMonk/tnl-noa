/***************************************************************************
                          tnl-view.h  -  description
                             -------------------
    begin                : Jan 21, 2013
    copyright            : (C) 2013 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#ifndef TNL_VIEW_H_
#define TNL_VIEW_H_

#include <cstdlib>
#include <TNL/core/mfilename.h>
#include <TNL/Config/ParameterContainer.h>
#include <TNL/String.h>
#include <TNL/Vectors/Vector.h>
#include <TNL/Vectors/MultiVector.h>
#include <TNL/mesh/tnlGrid.h>
#include <TNL/functions/tnlMeshFunction.h>

using namespace std;
using namespace TNL;

bool getOutputFileName( const String& inputFileName,
                        const String& outputFormat,
                        String& outputFileName )
{
   outputFileName = inputFileName;
   RemoveFileExtension( outputFileName );
   if( outputFormat == "gnuplot" )
   {
      outputFileName += ".gplt";
      return true;
   }
   if( outputFormat == "vtk" )
   {
      outputFileName += ".vtk";
      return true;
   }
   std::cerr << "Unknown file format " << outputFormat << ".";
   return false;
}


template< typename MeshFunction >
bool writeMeshFunction( const typename MeshFunction::MeshType& mesh,
                        const String& inputFileName,
                        const Config::ParameterContainer& parameters  )
{
   MeshFunction function( mesh );
   if( ! function.load( inputFileName ) )
   {
      std::cerr << "Unable to load mesh function from a file " << inputFileName << "." << std::endl;
      return false;
   }

   int verbose = parameters. getParameter< int >( "verbose");
   String outputFormat = parameters. getParameter< String >( "output-format" );
   String outputFileName;
   if( ! getOutputFileName( inputFileName,
                            outputFormat,
                            outputFileName ) )
      return false;
   if( verbose )
     std::cout << " writing to " << outputFileName << " ... " << std::flush;

   return function.write( outputFileName, outputFormat );
}

template< typename Mesh,
          int EntityDimensions,
          typename Real >
bool setMeshFunctionRealType( const Mesh& mesh,
                              const String& inputFileName,
                              const Config::ParameterContainer& parameters  )
{
   return writeMeshFunction< tnlMeshFunction< Mesh, EntityDimensions, Real > >( mesh, inputFileName, parameters );
}

template< typename Mesh,
          int EntityDimensions >
bool setMeshEntityType( const Mesh& mesh,
                        const String& inputFileName,
                        const List< String >& parsedObjectType,
                        const Config::ParameterContainer& parameters )
{
   if( parsedObjectType[ 3 ] == "float" )
      return setMeshFunctionRealType< Mesh, EntityDimensions, float >( mesh, inputFileName, parameters );
   if( parsedObjectType[ 3 ] == "double" )
      return setMeshFunctionRealType< Mesh, EntityDimensions, double >( mesh, inputFileName, parameters );
   if( parsedObjectType[ 3 ] == "long double" )
      return setMeshFunctionRealType< Mesh, EntityDimensions, long double >( mesh, inputFileName, parameters );
   std::cerr << "Unsupported arithmetics " << parsedObjectType[ 3 ] << " in mesh function " << inputFileName << std::endl;
   return false;
}

template< typename MeshReal,
          typename MeshIndex >
bool setMeshEntityDimensions( const tnlGrid< 1, MeshReal, tnlHost, MeshIndex >& mesh,
                              const String& inputFileName,
                              const List< String >& parsedObjectType,
                              const Config::ParameterContainer& parameters )
{
   typedef tnlGrid< 1, MeshReal, tnlHost, MeshIndex > Mesh;
   int meshEntityDimensions = atoi( parsedObjectType[ 2 ].getString() );
   switch( meshEntityDimensions )
   {
      case 0:
         return setMeshEntityType< Mesh, 0 >( mesh, inputFileName, parsedObjectType, parameters );
         break;
      case 1:
         return setMeshEntityType< Mesh, 1 >( mesh, inputFileName, parsedObjectType, parameters );
         break;
      default:
         std::cerr << "Unsupported mesh functions entity dimensions count " << meshEntityDimensions << "." << std::endl;
         return false;
   }
}

template< typename MeshReal,
          typename MeshIndex >
bool setMeshEntityDimensions( const tnlGrid< 2, MeshReal, tnlHost, MeshIndex >& mesh,
                              const String& inputFileName,
                              const List< String >& parsedObjectType,
                              const Config::ParameterContainer& parameters )
{
   typedef tnlGrid< 2, MeshReal, tnlHost, MeshIndex > Mesh;
   int meshEntityDimensions = atoi( parsedObjectType[ 2 ].getString() );
   switch( meshEntityDimensions )
   {
      case 0:
         return setMeshEntityType< Mesh, 0 >( mesh, inputFileName, parsedObjectType, parameters );
         break;
      case 1:
         return setMeshEntityType< Mesh, 1 >( mesh, inputFileName, parsedObjectType, parameters );
         break;
      case 2:
         return setMeshEntityType< Mesh, 2 >( mesh, inputFileName, parsedObjectType, parameters );
         break;
      default:
         std::cerr << "Unsupported mesh functions entity dimensions count " << meshEntityDimensions << "." << std::endl;
         return false;
   }
}

template< typename MeshReal,
          typename MeshIndex >
bool setMeshEntityDimensions( const tnlGrid< 3, MeshReal, tnlHost, MeshIndex >& mesh,
                              const String& inputFileName,
                              const List< String >& parsedObjectType,
                              const Config::ParameterContainer& parameters )
{
   typedef tnlGrid< 3, MeshReal, tnlHost, MeshIndex > Mesh;
   int meshEntityDimensions = atoi( parsedObjectType[ 2 ].getString() );
   switch( meshEntityDimensions )
   {
      case 0:
         return setMeshEntityType< Mesh, 0 >( mesh, inputFileName, parsedObjectType, parameters );
         break;
      case 1:
         return setMeshEntityType< Mesh, 1 >( mesh, inputFileName, parsedObjectType, parameters );
         break;
      case 2:
         return setMeshEntityType< Mesh, 2 >( mesh, inputFileName, parsedObjectType, parameters );
         break;
      case 3:
         return setMeshEntityType< Mesh, 3 >( mesh, inputFileName, parsedObjectType, parameters );
         break;
      default:
         std::cerr << "Unsupported mesh functions entity dimensions count " << meshEntityDimensions << "." << std::endl;
         return false;
   }
}

template< typename Mesh >
bool setMeshFunction( const Mesh& mesh,
                      const String& inputFileName,
                      const List< String >& parsedObjectType,
                      const Config::ParameterContainer& parameters )
{
   if( parsedObjectType[ 1 ] != mesh.getSerializationType() )
   {
      std::cerr << "Incompatible mesh type for the mesh function " << inputFileName << "." << std::endl;
      return false;
   }
   return setMeshEntityDimensions( mesh, inputFileName, parsedObjectType, parameters );
}


template< typename Mesh, typename Element, typename Real, typename Index, int Dimensions >
bool convertObject( const Mesh& mesh,
                    const String& inputFileName,
                    const List< String >& parsedObjectType,
                    const Config::ParameterContainer& parameters )
{
   int verbose = parameters. getParameter< int >( "verbose");
   String outputFormat = parameters. getParameter< String >( "output-format" );
   String outputFileName;
   if( ! getOutputFileName( inputFileName,
                            outputFormat,
                            outputFileName ) )
      return false;
   if( verbose )
     std::cout << " writing to " << outputFileName << " ... " << std::flush;


   if( parsedObjectType[ 0 ] == "tnlSharedVector" ||
       parsedObjectType[ 0 ] == "Vector" )
   {
      Vectors::Vector< Element, tnlHost, Index > vector;
      if( ! vector. load( inputFileName ) )
         return false;
      if( ! mesh. write( vector, outputFileName, outputFormat ) )
         return false;
   }

   if( parsedObjectType[ 0 ] == "tnlMultiVector" ||
       parsedObjectType[ 0 ] == "tnlSharedMultiVector" )
   {
      Vectors::tnlMultiVector< Dimensions, Element, tnlHost, Index > multiVector;
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

template< typename Mesh, typename Element, typename Real, typename Index >
bool setDimensions( const Mesh& mesh,
                    const String& inputFileName,
                    const List< String >& parsedObjectType,
                    const Config::ParameterContainer& parameters )
{
   int dimensions( 0 );
   if( parsedObjectType[ 0 ] == "tnlMultiVector" ||
       parsedObjectType[ 0 ] == "tnlSharedMultiVector" )
      dimensions = atoi( parsedObjectType[ 1 ]. getString() );
   if( parsedObjectType[ 0 ] == "Vector" ||
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
   std::cerr << "Cannot convert objects with " << dimensions << " dimensions." << std::endl;
   return false;
}

template< typename Mesh, typename Element, typename Real >
bool setIndexType( const Mesh& mesh,
                   const String& inputFileName,
                   const List< String >& parsedObjectType,
                   const Config::ParameterContainer& parameters )
{
   String indexType;
   if( parsedObjectType[ 0 ] == "tnlMultiVector" ||
       parsedObjectType[ 0 ] == "tnlSharedMultiVector" )
      indexType = parsedObjectType[ 4 ];
   if( parsedObjectType[ 0 ] == "tnlSharedVector" ||
       parsedObjectType[ 0 ] == "Vector" )
      indexType = parsedObjectType[ 3 ];

   if( indexType == "int" )
      return setDimensions< Mesh, Element, Real, int >( mesh, inputFileName, parsedObjectType, parameters );
   if( indexType == "long-int" )
      return setDimensions< Mesh, Element, Real, long int >( mesh, inputFileName, parsedObjectType, parameters );
   std::cerr << "Unknown index type " << indexType << "." << std::endl;
   return false;
}

template< typename Mesh >
bool setTupleType( const Mesh& mesh,
                   const String& inputFileName,
                   const List< String >& parsedObjectType,
                   const List< String >& parsedElementType,
                   const Config::ParameterContainer& parameters )
{
   int dimensions = atoi( parsedElementType[ 1 ].getString() );
   String dataType = parsedElementType[ 2 ];
   if( dataType == "float" )
      switch( dimensions )
      {
         case 1:
            return setIndexType< Mesh, Vectors::StaticVector< 1, float >, float >( mesh, inputFileName, parsedObjectType, parameters );
            break;
         case 2:
            return setIndexType< Mesh, Vectors::StaticVector< 2, float >, float >( mesh, inputFileName, parsedObjectType, parameters );
            break;
         case 3:
            return setIndexType< Mesh, Vectors::StaticVector< 3, float >, float >( mesh, inputFileName, parsedObjectType, parameters );
            break;
      }
   if( dataType == "double" )
      switch( dimensions )
      {
         case 1:
            return setIndexType< Mesh, Vectors::StaticVector< 1, double >, double >( mesh, inputFileName, parsedObjectType, parameters );
            break;
         case 2:
            return setIndexType< Mesh, Vectors::StaticVector< 2, double >, double >( mesh, inputFileName, parsedObjectType, parameters );
            break;
         case 3:
            return setIndexType< Mesh, Vectors::StaticVector< 3, double >, double >( mesh, inputFileName, parsedObjectType, parameters );
            break;
      }
   if( dataType == "long double" )
      switch( dimensions )
      {
         case 1:
            return setIndexType< Mesh, Vectors::StaticVector< 1, long double >, long double >( mesh, inputFileName, parsedObjectType, parameters );
            break;
         case 2:
            return setIndexType< Mesh, Vectors::StaticVector< 2, long double >, long double >( mesh, inputFileName, parsedObjectType, parameters );
            break;
         case 3:
            return setIndexType< Mesh, Vectors::StaticVector< 3, long double >, long double >( mesh, inputFileName, parsedObjectType, parameters );
            break;
      }
}

template< typename Mesh >
bool setElementType( const Mesh& mesh,
                     const String& inputFileName,
                     const List< String >& parsedObjectType,
                     const Config::ParameterContainer& parameters )
{
   String elementType;

   // TODO: Fix this even for arrays
   if( parsedObjectType[ 0 ] == "tnlMultiVector" ||
       parsedObjectType[ 0 ] == "tnlSharedMultiVector" )
      elementType = parsedObjectType[ 2 ];
   if( parsedObjectType[ 0 ] == "tnlSharedVector" ||
       parsedObjectType[ 0 ] == "Vector" )
      elementType = parsedObjectType[ 1 ];


   if( elementType == "float" )
      return setIndexType< Mesh, float, float >( mesh, inputFileName, parsedObjectType, parameters );
   if( elementType == "double" )
      return setIndexType< Mesh, double, double >( mesh, inputFileName, parsedObjectType, parameters );
   if( elementType == "long double" )
      return setIndexType< Mesh, long double, long double >( mesh, inputFileName, parsedObjectType, parameters );
   List< String > parsedElementType;
   if( ! parseObjectType( elementType, parsedElementType ) )
   {
      std::cerr << "Unable to parse object type " << elementType << "." << std::endl;
      return false;
   }
   if( parsedElementType[ 0 ] == "StaticVector" )
      return setTupleType< Mesh >( mesh, inputFileName, parsedObjectType, parsedElementType, parameters );

   std::cerr << "Unknown element type " << elementType << "." << std::endl;
   return false;
}

template< typename Mesh >
bool processFiles( const Config::ParameterContainer& parameters )
{
   int verbose = parameters. getParameter< int >( "verbose");
   String meshFile = parameters. getParameter< String >( "mesh" );

   Mesh mesh;
   if( meshFile != "" )
      if( ! mesh. load( meshFile ) )
      {
         std::cerr << "I am not able to load mesh from the file " << meshFile << "." << std::endl;
         return false;
      }
   mesh. writeMesh( "mesh.asy", "asymptote" );

   bool checkOutputFile = parameters. getParameter< bool >( "check-output-file" );
   List< String > inputFiles = parameters. getParameter< List< String > >( "input-files" );
   bool error( false );
//#ifdef HAVE_OPENMP
//#pragma omp parallel for
//#endif
   for( int i = 0; i < inputFiles. getSize(); i ++ )
   {
      if( verbose )
        std::cout << "Processing file " << inputFiles[ i ] << " ... " << std::flush;

      String outputFormat = parameters. getParameter< String >( "output-format" );
      String outputFileName;
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
           std::cout << " file already exists. Skipping.            \r" << std::flush;
         continue;
      }

      String objectType;
      if( ! getObjectType( inputFiles[ i ], objectType ) )
          std::cerr << "unknown object ... SKIPPING!" << std::endl;
      else
      {
         if( verbose )
           std::cout << objectType << " detected ... ";

         List< String > parsedObjectType;
         if( ! parseObjectType( objectType, parsedObjectType ) )
         {
            std::cerr << "Unable to parse object type " << objectType << "." << std::endl;
            error = true;
            continue;
         }
         if( parsedObjectType[ 0 ] == "tnlMultiVector" ||
             parsedObjectType[ 0 ] == "tnlSharedMultiVector" ||
             parsedObjectType[ 0 ] == "tnlSharedVector" ||
             parsedObjectType[ 0 ] == "Vector" )
            setElementType< Mesh >( mesh, inputFiles[ i ], parsedObjectType, parameters );
         if( parsedObjectType[ 0 ] == "tnlMeshFunction" )
            setMeshFunction< Mesh >( mesh, inputFiles[ i ], parsedObjectType, parameters );
         if( verbose )
           std::cout << "[ OK ].  " << std::endl;

      }
   }
   if( verbose )
     std::cout << std::endl;
   return ! error;
}


#endif /* TNL_VIEW_H_ */
