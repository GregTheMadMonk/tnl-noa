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
#include <TNL/Functions/VectorField.h>

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
   double scale = parameters. getParameter< double >( "scale" );
   String outputFileName;
   if( ! getOutputFileName( inputFileName,
                            outputFormat,
                            outputFileName ) )
      return false;
   if( verbose )
     std::cout << " writing to " << outputFileName << " ... " << std::flush;

   return function.write( outputFileName, outputFormat, scale );
}

template< typename VectorField >
bool writeVectorField( const typename VectorField::FunctionType::MeshPointer& meshPointer,
                       const String& inputFileName,
                       const Config::ParameterContainer& parameters  )
{

   VectorField field( meshPointer );
   std::cout << "VectorField: " << field.getType() << std::endl;
   if( ! field.load( inputFileName ) )
   {
      std::cerr << "Unable to load vector field from a file " << inputFileName << "." << std::endl;
      return false;
   }

   int verbose = parameters. getParameter< int >( "verbose");
   String outputFormat = parameters. getParameter< String >( "output-format" );
   double scale = parameters. getParameter< double >( "scale" );
   String outputFileName;
   if( ! getOutputFileName( inputFileName,
                            outputFormat,
                            outputFileName ) )
      return false;
   if( verbose )
     std::cout << " writing to " << outputFileName << " ... " << std::flush;

   return field.write( outputFileName, outputFormat, scale );
}


template< typename MeshPointer,
          int EntityDimension,
          typename Real,
          int VectorFieldSize >
bool setMeshFunctionRealType( const MeshPointer& meshPointer,
                              const String& inputFileName,
                              const Containers::List< String >& parsedObjectType,
                              const Config::ParameterContainer& parameters  )
{
   if( VectorFieldSize == 0 )
      return writeMeshFunction< Functions::MeshFunction< typename MeshPointer::ObjectType, EntityDimension, Real > >( meshPointer, inputFileName, parameters );
   return writeVectorField< Functions::VectorField< VectorFieldSize, Functions::MeshFunction< typename MeshPointer::ObjectType, EntityDimension, Real > > >( meshPointer, inputFileName, parameters );
}

template< typename MeshPointer,
          int EntityDimension,
          int VectorFieldSize >
bool setMeshEntityType( const MeshPointer& meshPointer,
                        const String& inputFileName,
                        const Containers::List< String >& parsedObjectType,
                        const Config::ParameterContainer& parameters )
{
   if( parsedObjectType[ 3 ] == "float" )
      return setMeshFunctionRealType< MeshPointer, EntityDimension, float, VectorFieldSize >( meshPointer, inputFileName, parsedObjectType, parameters );
   if( parsedObjectType[ 3 ] == "double" )
      return setMeshFunctionRealType< MeshPointer, EntityDimension, double, VectorFieldSize >( meshPointer, inputFileName, parsedObjectType, parameters );
   if( parsedObjectType[ 3 ] == "long double" )
      return setMeshFunctionRealType< MeshPointer, EntityDimension, long double, VectorFieldSize >( meshPointer, inputFileName, parsedObjectType, parameters );
   std::cerr << "Unsupported arithmetics " << parsedObjectType[ 3 ] << " in mesh function " << inputFileName << std::endl;
   return false;
}

template< typename MeshReal,
          typename MeshIndex,
          int VectorFieldSize >
bool setMeshEntityDimension( const SharedPointer< Meshes::Grid< 1, MeshReal, Devices::Host, MeshIndex > >& meshPointer,
                              const String& inputFileName,
                              const Containers::List< String >& parsedObjectType,
                              const Config::ParameterContainer& parameters )
{
   typedef Meshes::Grid< 1, MeshReal, Devices::Host, MeshIndex > Mesh;
   typedef SharedPointer< Mesh > MeshPointer;
   int meshEntityDimension = atoi( parsedObjectType[ 2 ].getString() );
   switch( meshEntityDimension )
   {
      case 0:
         return setMeshEntityType< MeshPointer, 0, VectorFieldSize >( meshPointer, inputFileName, parsedObjectType, parameters );
         break;      
      case 1:
         return setMeshEntityType< MeshPointer, 1, VectorFieldSize >( meshPointer, inputFileName, parsedObjectType, parameters );
         break;
      default:
         std::cerr << "Unsupported mesh functions entity dimension count " << meshEntityDimension << "." << std::endl;
         return false;
   }
}

template< typename MeshReal,
          typename MeshIndex,
          int VectorFieldSize >
bool setMeshEntityDimension( const SharedPointer< Meshes::Grid< 2, MeshReal, Devices::Host, MeshIndex > >& meshPointer,
                              const String& inputFileName,
                              const Containers::List< String >& parsedObjectType,
                              const Config::ParameterContainer& parameters )
{
   typedef Meshes::Grid< 2, MeshReal, Devices::Host, MeshIndex > Mesh;
   typedef SharedPointer< Mesh > MeshPointer;
   int meshEntityDimension = atoi( parsedObjectType[ 2 ].getString() );
   switch( meshEntityDimension )
   {
      case 0:
         return setMeshEntityType< MeshPointer, 0, VectorFieldSize >( meshPointer, inputFileName, parsedObjectType, parameters );
         break;            
      case 1:
         return setMeshEntityType< MeshPointer, 1, VectorFieldSize >( meshPointer, inputFileName, parsedObjectType, parameters );
         break;
      case 2:
         return setMeshEntityType< MeshPointer, 2, VectorFieldSize >( meshPointer, inputFileName, parsedObjectType, parameters );
         break;
      default:
         std::cerr << "Unsupported mesh functions entity dimension count " << meshEntityDimension << "." << std::endl;
         return false;
   }
}

template< typename MeshReal,
          typename MeshIndex,
          int VectorFieldSize >
bool setMeshEntityDimension( const SharedPointer< Meshes::Grid< 3, MeshReal, Devices::Host, MeshIndex > >& meshPointer,
                              const String& inputFileName,
                              const Containers::List< String >& parsedObjectType,
                              const Config::ParameterContainer& parameters )
{
   typedef Meshes::Grid< 3, MeshReal, Devices::Host, MeshIndex > Mesh;
   typedef SharedPointer< Mesh > MeshPointer;
   int meshEntityDimension = atoi( parsedObjectType[ 2 ].getString() );
   switch( meshEntityDimension )
   {
      case 0:
         return setMeshEntityType< MeshPointer, 0, VectorFieldSize >( meshPointer, inputFileName, parsedObjectType, parameters );
         break;      
      case 1:
         return setMeshEntityType< MeshPointer, 1, VectorFieldSize >( meshPointer, inputFileName, parsedObjectType, parameters );
         break;
      case 2:
         return setMeshEntityType< MeshPointer, 2, VectorFieldSize >( meshPointer, inputFileName, parsedObjectType, parameters );
         break;
      case 3:
         return setMeshEntityType< MeshPointer, 3, VectorFieldSize >( meshPointer, inputFileName, parsedObjectType, parameters );
         break;
      default:
         std::cerr << "Unsupported mesh functions entity dimension count " << meshEntityDimension << "." << std::endl;
         return false;
   }
}

template< typename MeshPointer, int VectorFieldSize = 0 >
bool setMeshFunction( const MeshPointer& meshPointer,
                      const String& inputFileName,
                      const Containers::List< String >& parsedObjectType,
                      const Config::ParameterContainer& parameters )
{
   std::cerr << parsedObjectType[ 1 ] << std::endl;
   if( parsedObjectType[ 1 ] != meshPointer->getSerializationType() )
   {
      std::cerr << "Incompatible mesh type for the mesh function " << inputFileName << "." << std::endl;
      return false;
   }
   typedef typename MeshPointer::ObjectType::RealType RealType;
   typedef typename MeshPointer::ObjectType::IndexType IndexType;
   return setMeshEntityDimension< RealType, IndexType, VectorFieldSize >( meshPointer, inputFileName, parsedObjectType, parameters );
}

template< typename MeshPointer >
bool setVectorFieldSize( const MeshPointer& meshPointer,
                         const String& inputFileName,
                         Containers::List< String >& parsedObjectType,
                         const Config::ParameterContainer& parameters )
{
   int vectorFieldSize = atoi( parsedObjectType[ 1 ].getString() );
   Containers::List< String > parsedMeshFunctionType;
   if( ! parseObjectType( parsedObjectType[ 2 ], parsedMeshFunctionType ) )
   {
      std::cerr << "Unable to parse mesh function type  " << parsedObjectType[ 2 ] << " in a vector field." << std::endl;
      return false;
   }
   switch( vectorFieldSize )
   {
      case 1:
         return setMeshFunction< MeshPointer, 1 >( meshPointer, inputFileName, parsedMeshFunctionType, parameters );
      case 2:
         return setMeshFunction< MeshPointer, 2 >( meshPointer, inputFileName, parsedMeshFunctionType, parameters );
      case 3:
         return setMeshFunction< MeshPointer, 3 >( meshPointer, inputFileName, parsedMeshFunctionType, parameters );
   }
   std::cerr << "Unsupported vector field size " << vectorFieldSize << "." << std::endl;
   return false;
}

template< typename MeshPointer, typename Element, typename Real, typename Index, int Dimension >
bool convertObject( const MeshPointer& meshPointer,
                    const String& inputFileName,
                    const Containers::List< String >& parsedObjectType,
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
       parsedObjectType[ 0 ] == "TNL::Containers::Vector" )          //
   {
      using MeshType = typename MeshPointer::ObjectType;
      // FIXME: why is MeshType::GlobalIndexType not the same as Index?
//      Containers::Vector< Element, Devices::Host, Index > vector;
      Containers::Vector< Element, Devices::Host, typename MeshType::GlobalIndexType > vector;
      if( ! vector.load( inputFileName ) )
         return false;
      Functions::MeshFunction< MeshType, MeshType::getMeshDimension(), Element > mf;
      mf.bind( meshPointer, vector );
      if( ! mf.write( outputFileName, outputFormat ) )
         return false;
   }

   if( parsedObjectType[ 0 ] == "Containers::MultiVector" ||
       parsedObjectType[ 0 ] == "TNL::Containers::MultiVector" ||      // TODO: remove deprecated type names  
       parsedObjectType[ 0 ] == "tnlSharedMultiVector" ) //
   {
      Containers::MultiVector< Dimension, Element, Devices::Host, Index > multiVector;
      if( ! multiVector. load( inputFileName ) )
         return false;
      typedef Meshes::Grid< Dimension, Real, Devices::Host, Index > GridType;
      typedef typename GridType::PointType PointType;
      typedef typename GridType::CoordinatesType CoordinatesType;
      GridType grid;
      grid. setDomain( PointType( 0.0 ), PointType( 1.0 ) );
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
                    const Containers::List< String >& parsedObjectType,
                    const Config::ParameterContainer& parameters )
{
   int dimensions( 0 );
   if( parsedObjectType[ 0 ] == "Containers::MultiVector" ||
       parsedObjectType[ 0 ] == "TNL::Containers::MultiVector" ||                     // TODO: remove deprecated type names
       parsedObjectType[ 0 ] == "tnlSharedMultiVector" )                //
      dimensions = atoi( parsedObjectType[ 1 ]. getString() );
   if( parsedObjectType[ 0 ] == "Containers::Vector" ||
       parsedObjectType[ 0 ] == "TNL::Containers::Vector" ||                          // TODO: remove deprecated type names
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
                   const Containers::List< String >& parsedObjectType,
                   const Config::ParameterContainer& parameters )
{
   String indexType;
   if( parsedObjectType[ 0 ] == "Containers::MultiVector" ||
       parsedObjectType[ 0 ] == "TNL::Containers::MultiVector" ||                        // TODO: remove deprecated type names
       parsedObjectType[ 0 ] == "tnlSharedMultiVector" )                   //
      indexType = parsedObjectType[ 4 ];
   if( parsedObjectType[ 0 ] == "Containers::Vector" || 
       parsedObjectType[ 0 ] == "tnlSharedVector" ||                       // TODO: remove deprecated type names
       parsedObjectType[ 0 ] == "TNL::Containers::Vector" )                              //
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
                   const Containers::List< String >& parsedObjectType,
                   const Containers::List< String >& parsedElementType,
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
                     const Containers::List< String >& parsedObjectType,
                     const Config::ParameterContainer& parameters )
{
   String elementType;

   // TODO: Fix this even for arrays
   if( parsedObjectType[ 0 ] == "Containers::MultiVector" ||
       parsedObjectType[ 0 ] == "TNL::Containers::MultiVector" ||                            // TODO: remove deprecated type names
       parsedObjectType[ 0 ] == "tnlSharedMultiVector" )                       //
      elementType = parsedObjectType[ 2 ];
   if( parsedObjectType[ 0 ] == "Containers::Vector" ||
       parsedObjectType[ 0 ] == "tnlSharedVector" ||                           // TODO: remove deprecated type names
       parsedObjectType[ 0 ] == "TNL::Containers::Vector" )                                  //
      elementType = parsedObjectType[ 1 ];


   if( elementType == "float" )
      return setIndexType< MeshPointer, float, float >( meshPointer, inputFileName, parsedObjectType, parameters );
   if( elementType == "double" )
      return setIndexType< MeshPointer, double, double >( meshPointer, inputFileName, parsedObjectType, parameters );
   if( elementType == "long double" )
      return setIndexType< MeshPointer, long double, long double >( meshPointer, inputFileName, parsedObjectType, parameters );
   Containers::List< String > parsedElementType;
   if( ! parseObjectType( elementType, parsedElementType ) )
   {
      std::cerr << "Unable to parse object type " << elementType << "." << std::endl;
      return false;
   }
   if( parsedElementType[ 0 ] == "Containers::StaticVector" ||
       parsedElementType[ 0 ] == "Containers::StaticVector" )               // TODO: remove deprecated type names
      return setTupleType< MeshPointer >( meshPointer, inputFileName, parsedObjectType, parsedElementType, parameters );

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
   Containers::List< String > inputFiles = parameters. getParameter< Containers::List< String > >( "input-files" );
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

         Containers::List< String > parsedObjectType;
         if( ! parseObjectType( objectType, parsedObjectType ) )
         {
            std::cerr << "Unable to parse object type " << objectType << "." << std::endl;
            error = true;
            continue;
         }
         if( parsedObjectType[ 0 ] == "Containers::MultiVector" ||
             parsedObjectType[ 0 ] == "Containers::Vector" ||
             parsedObjectType[ 0 ] == "TNL::Containers::MultiVector" ||                     // TODO: remove deprecated type names
             parsedObjectType[ 0 ] == "tnlSharedMultiVector" ||               // 
             parsedObjectType[ 0 ] == "tnlSharedVector" ||                    //
             parsedObjectType[ 0 ] == "TNL::Containers::Vector" )                           //
            setElementType< MeshPointer >( meshPointer, inputFiles[ i ], parsedObjectType, parameters );
         if( parsedObjectType[ 0 ] == "Functions::MeshFunction" ||
             parsedObjectType[ 0 ] == "tnlMeshFunction" )                     // TODO: remove deprecated type names
            setMeshFunction< MeshPointer >( meshPointer, inputFiles[ i ], parsedObjectType, parameters );
         if( parsedObjectType[ 0 ] == "Functions::VectorField" )
            setVectorFieldSize< MeshPointer >( meshPointer, inputFiles[ i ], parsedObjectType, parameters );
         if( verbose )
           std::cout << "[ OK ].  " << std::endl;
      }
   }
   if( verbose )
     std::cout << std::endl;
   return ! error;
}

#endif /* TNL_VIEW_H_ */
