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
#include <TNL/FileName.h>
#include <TNL/Config/ParameterContainer.h>
#include <TNL/String.h>
#include <TNL/Containers/Vector.h>
#include <TNL/Containers/MultiVector.h>
#include <TNL/Meshes/Grid.h>
#include <TNL/Functions/MeshFunction.h>

using namespace TNL;

bool getOutputFileName( const String& inputFileName,
                        const String& outputFormat,
                        String& outputFileName )
{
   outputFileName = inputFileName;
   removeFileExtension( outputFileName );
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
bool writeMeshFunction( const typename MeshFunction::MeshPointer& meshPointer,
                        const String& inputFileName,
                        const Config::ParameterContainer& parameters  )
{
   MeshFunction function( meshPointer );
   std::cout << "Mesh function: " << function.getType() << std::endl;
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

template< typename MeshPointer,
          int EntityDimensions,
          typename Real >
bool setMeshFunctionRealType( const MeshPointer& meshPointer,
                              const String& inputFileName,
                              const Config::ParameterContainer& parameters  )
{
   return writeMeshFunction< Functions::MeshFunction< typename MeshPointer::ObjectType, EntityDimensions, Real > >( meshPointer, inputFileName, parameters );
}

template< typename MeshPointer,
          int EntityDimensions >
bool setMeshEntityType( const MeshPointer& meshPointer,
                        const String& inputFileName,
                        const List< String >& parsedObjectType,
                        const Config::ParameterContainer& parameters )
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
bool setMeshEntityDimensions( const SharedPointer< Meshes::Grid< 1, MeshReal, Devices::Host, MeshIndex > >& meshPointer,
                              const String& inputFileName,
                              const List< String >& parsedObjectType,
                              const Config::ParameterContainer& parameters )
{
   typedef Meshes::Grid< 1, MeshReal, Devices::Host, MeshIndex > Mesh;
   typedef SharedPointer< Mesh > MeshPointer;
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
         std::cerr << "Unsupported mesh functions entity dimensions count " << meshEntityDimensions << "." << std::endl;
         return false;
   }
}

template< typename MeshReal,
          typename MeshIndex >
bool setMeshEntityDimensions( const SharedPointer< Meshes::Grid< 2, MeshReal, Devices::Host, MeshIndex > >& meshPointer,
                              const String& inputFileName,
                              const List< String >& parsedObjectType,
                              const Config::ParameterContainer& parameters )
{
   typedef Meshes::Grid< 2, MeshReal, Devices::Host, MeshIndex > Mesh;
   typedef SharedPointer< Mesh > MeshPointer;
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
         std::cerr << "Unsupported mesh functions entity dimensions count " << meshEntityDimensions << "." << std::endl;
         return false;
   }
}

template< typename MeshReal,
          typename MeshIndex >
bool setMeshEntityDimensions( const SharedPointer< Meshes::Grid< 3, MeshReal, Devices::Host, MeshIndex > >& meshPointer,
                              const String& inputFileName,
                              const List< String >& parsedObjectType,
                              const Config::ParameterContainer& parameters )
{
   typedef Meshes::Grid< 3, MeshReal, Devices::Host, MeshIndex > Mesh;
   typedef SharedPointer< Mesh > MeshPointer;
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
         std::cerr << "Unsupported mesh functions entity dimensions count " << meshEntityDimensions << "." << std::endl;
         return false;
   }
}

template< typename MeshPointer >
bool setMeshFunction( const MeshPointer& meshPointer,
                      const String& inputFileName,
                      const List< String >& parsedObjectType,
                      const Config::ParameterContainer& parameters )
{
   if( parsedObjectType[ 1 ] != meshPointer->getSerializationType() )
   {
      std::cerr << "Incompatible mesh type for the mesh function " << inputFileName << "." << std::endl;
      return false;
   }
   return setMeshEntityDimensions( meshPointer, inputFileName, parsedObjectType, parameters );
}


template< typename MeshPointer, typename Element, typename Real, typename Index, int Dimensions >
bool convertObject( const MeshPointer& meshPointer,
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


   if( parsedObjectType[ 0 ] == "Containers::Vector" ||
       parsedObjectType[ 0 ] == "tnlSharedVector" ||   // TODO: remove deprecated type names
       parsedObjectType[ 0 ] == "tnlVector" )          //
   {
      using MeshType = typename MeshPointer::ObjectType;
      // FIXME: why is MeshType::IndexType not the same as Index?
//      Containers::Vector< Element, Devices::Host, Index > vector;
      Containers::Vector< Element, Devices::Host, typename MeshType::IndexType > vector;
      if( ! vector.load( inputFileName ) )
         return false;
      Functions::MeshFunction< MeshType, MeshType::meshDimensions, Element > mf;
      mf.bind( meshPointer, vector );
      if( ! mf.write( outputFileName, outputFormat ) )
         return false;
   }

   if( parsedObjectType[ 0 ] == "Containers::MultiVector" ||
       parsedObjectType[ 0 ] == "tnlMultiVector" ||      // TODO: remove deprecated type names  
       parsedObjectType[ 0 ] == "tnlSharedMultiVector" ) //
   {
      Containers::MultiVector< Dimensions, Element, Devices::Host, Index > multiVector;
      if( ! multiVector. load( inputFileName ) )
         return false;
      typedef Meshes::Grid< Dimensions, Real, Devices::Host, Index > GridType;
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
                    const String& inputFileName,
                    const List< String >& parsedObjectType,
                    const Config::ParameterContainer& parameters )
{
   int dimensions( 0 );
   if( parsedObjectType[ 0 ] == "Containers::MultiVector" ||
       parsedObjectType[ 0 ] == "tnlMultiVector" ||                     // TODO: remove deprecated type names
       parsedObjectType[ 0 ] == "tnlSharedMultiVector" )                //
      dimensions = atoi( parsedObjectType[ 1 ]. getString() );
   if( parsedObjectType[ 0 ] == "Containers::Vector" ||
       parsedObjectType[ 0 ] == "tnlVector" ||                          // TODO: remove deprecated type names
       parsedObjectType[ 0 ] == "tnlSharedVector" )                     //
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
   std::cerr << "Cannot convert objects with " << dimensions << " dimensions." << std::endl;
   return false;
}

template< typename MeshPointer, typename Element, typename Real >
bool setIndexType( const MeshPointer& meshPointer,
                   const String& inputFileName,
                   const List< String >& parsedObjectType,
                   const Config::ParameterContainer& parameters )
{
   String indexType;
   if( parsedObjectType[ 0 ] == "Containers::MultiVector" ||
       parsedObjectType[ 0 ] == "tnlMultiVector" ||                        // TODO: remove deprecated type names
       parsedObjectType[ 0 ] == "tnlSharedMultiVector" )                   //
      indexType = parsedObjectType[ 4 ];
   if( parsedObjectType[ 0 ] == "Containers::Vector" || 
       parsedObjectType[ 0 ] == "tnlSharedVector" ||                       // TODO: remove deprecated type names
       parsedObjectType[ 0 ] == "tnlVector" )                              //
      indexType = parsedObjectType[ 3 ];

   if( indexType == "int" )
      return setDimensions< MeshPointer, Element, Real, int >( meshPointer, inputFileName, parsedObjectType, parameters );
   if( indexType == "long-int" )
      return setDimensions< MeshPointer, Element, Real, long int >( meshPointer, inputFileName, parsedObjectType, parameters );
   std::cerr << "Unknown index type " << indexType << "." << std::endl;
   return false;
}

template< typename MeshPointer >
bool setTupleType( const MeshPointer& meshPointer,
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
            return setIndexType< MeshPointer, Containers::StaticVector< 1, float >, float >( meshPointer, inputFileName, parsedObjectType, parameters );
            break;
         case 2:
            return setIndexType< MeshPointer, Containers::StaticVector< 2, float >, float >( meshPointer, inputFileName, parsedObjectType, parameters );
            break;
         case 3:
            return setIndexType< MeshPointer, Containers::StaticVector< 3, float >, float >( meshPointer, inputFileName, parsedObjectType, parameters );
            break;
      }
   if( dataType == "double" )
      switch( dimensions )
      {
         case 1:
            return setIndexType< MeshPointer, Containers::StaticVector< 1, double >, double >( meshPointer, inputFileName, parsedObjectType, parameters );
            break;
         case 2:
            return setIndexType< MeshPointer, Containers::StaticVector< 2, double >, double >( meshPointer, inputFileName, parsedObjectType, parameters );
            break;
         case 3:
            return setIndexType< MeshPointer, Containers::StaticVector< 3, double >, double >( meshPointer, inputFileName, parsedObjectType, parameters );
            break;
      }
   if( dataType == "long double" )
      switch( dimensions )
      {
         case 1:
            return setIndexType< MeshPointer, Containers::StaticVector< 1, long double >, long double >( meshPointer, inputFileName, parsedObjectType, parameters );
            break;
         case 2:
            return setIndexType< MeshPointer, Containers::StaticVector< 2, long double >, long double >( meshPointer, inputFileName, parsedObjectType, parameters );
            break;
         case 3:
            return setIndexType< MeshPointer, Containers::StaticVector< 3, long double >, long double >( meshPointer, inputFileName, parsedObjectType, parameters );
            break;
      }
   return false;
}

template< typename MeshPointer >
bool setElementType( const MeshPointer& meshPointer,
                     const String& inputFileName,
                     const List< String >& parsedObjectType,
                     const Config::ParameterContainer& parameters )
{
   String elementType;

   // TODO: Fix this even for arrays
   if( parsedObjectType[ 0 ] == "Containers::MultiVector" ||
       parsedObjectType[ 0 ] == "tnlMultiVector" ||                            // TODO: remove deprecated type names
       parsedObjectType[ 0 ] == "tnlSharedMultiVector" )                       //
      elementType = parsedObjectType[ 2 ];
   if( parsedObjectType[ 0 ] == "Containers::Vector" ||
       parsedObjectType[ 0 ] == "tnlSharedVector" ||                           // TODO: remove deprecated type names
       parsedObjectType[ 0 ] == "tnlVector" )                                  //
      elementType = parsedObjectType[ 1 ];


   if( elementType == "float" )
      return setIndexType< MeshPointer, float, float >( meshPointer, inputFileName, parsedObjectType, parameters );
   if( elementType == "double" )
      return setIndexType< MeshPointer, double, double >( meshPointer, inputFileName, parsedObjectType, parameters );
   if( elementType == "long double" )
      return setIndexType< MeshPointer, long double, long double >( meshPointer, inputFileName, parsedObjectType, parameters );
   List< String > parsedElementType;
   if( ! parseObjectType( elementType, parsedElementType ) )
   {
      std::cerr << "Unable to parse object type " << elementType << "." << std::endl;
      return false;
   }
   // FIXME: this does not compile for an unknown reason
//   if( parsedElementType[ 0 ] == "Containers::StaticVector" ||
//       parsedElementType[ 0 ] == "tnlStaticVector" )               // TODO: remove deprecated type names
//      return setTupleType< MeshPointer >( meshPointer, inputFileName, parsedObjectType, parsedElementType, parameters );

   std::cerr << "Unknown element type " << elementType << "." << std::endl;
   return false;
}

template< typename Mesh >
bool processFiles( const Config::ParameterContainer& parameters )
{
   int verbose = parameters. getParameter< int >( "verbose");
   String meshFile = parameters. getParameter< String >( "mesh" );

   typedef SharedPointer< Mesh > MeshPointer;
   MeshPointer meshPointer;
   
   if( meshFile != "" )
      if( ! meshPointer->load( meshFile ) )
      {
         std::cerr << "I am not able to load mesh from the file " << meshFile << "." << std::endl;
         return false;
      }
   meshPointer->writeMesh( "mesh.asy", "asymptote" );

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
         if( parsedObjectType[ 0 ] == "Containers::MultiVector" ||
             parsedObjectType[ 0 ] == "Containers::Vector" ||
             parsedObjectType[ 0 ] == "tnlMultiVector" ||                     // TODO: remove deprecated type names
             parsedObjectType[ 0 ] == "tnlSharedMultiVector" ||               // 
             parsedObjectType[ 0 ] == "tnlSharedVector" ||                    //
             parsedObjectType[ 0 ] == "tnlVector" )                           //
            setElementType< MeshPointer >( meshPointer, inputFiles[ i ], parsedObjectType, parameters );
         if( parsedObjectType[ 0 ] == "Functions::MeshFunction" ||
             parsedObjectType[ 0 ] == "tnlMeshFunction" )                     // TODO: remove deprecated type names
            setMeshFunction< MeshPointer >( meshPointer, inputFiles[ i ], parsedObjectType, parameters );
         if( verbose )
           std::cout << "[ OK ].  " << std::endl;
      }
   }
   if( verbose )
     std::cout << std::endl;
   return ! error;
}

#endif /* TNL_VIEW_H_ */
