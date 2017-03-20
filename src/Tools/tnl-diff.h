/***************************************************************************
                          tnl-diff.h  -  description
                             -------------------
    begin                : Nov 17, 2013
    copyright            : (C) 2013 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#ifndef TNL_DIFF_H_
#define TNL_DIFF_H_

#include <iomanip>
#include <TNL/Config/ParameterContainer.h>
#include <TNL/FileName.h>
#include <TNL/Containers/Vector.h>
#include <TNL/Containers/StaticVector.h>
#include <TNL/Functions/MeshFunction.h>

using namespace TNL;

template< typename MeshPointer, typename Element, typename Real, typename Index >
bool computeDifferenceOfMeshFunctions( const MeshPointer& meshPointer, const Config::ParameterContainer& parameters )
{
   bool verbose = parameters. getParameter< bool >( "verbose" );
   Containers::List< String > inputFiles = parameters. getParameter< Containers::List< String > >( "input-files" );
   String mode = parameters. getParameter< String >( "mode" );
   String outputFileName = parameters. getParameter< String >( "output-file" );
   double snapshotPeriod = parameters. getParameter< double >( "snapshot-period" );
   bool writeDifference = parameters. getParameter< bool >( "write-difference" );

   std::fstream outputFile;
   outputFile.open( outputFileName.getString(), std::fstream::out );
   if( ! outputFile )
   {
      std::cerr << "Unable to open the file " << outputFileName << "." << std::endl;
      return false;
   }
   outputFile << "#";
   outputFile << std::setw( 6 ) << "Time";
   outputFile << std::setw( 18 ) << "L1 diff."
              << std::setw( 18 ) << "L2 diff."
              << std::setw( 18 ) << "Max. diff."
              << std::setw( 18 ) << "Total L1 diff."
              << std::setw( 18 ) << "Total L2 diff."
              << std::setw( 18 ) << "Total Max. diff."
              << std::endl;
   if( verbose )
      std::cout << std::endl;
   
   typedef typename MeshPointer::ObjectType Mesh;
   Functions::MeshFunction< Mesh, Mesh::getMeshDimensions(), Real > v1( meshPointer ), v2( meshPointer ), diff( meshPointer );
   Real totalL1Diff( 0.0 ), totalL2Diff( 0.0 ), totalMaxDiff( 0.0 );
   for( int i = 0; i < inputFiles. getSize(); i ++ )
   {
      if( mode == "couples" )
      {
         if( i + 1 == inputFiles.getSize() )
         {
            std::cerr << std::endl << "Skipping the file " << inputFiles[ i ] << " since there is no file to couple it with." << std::endl;
            outputFile.close();
            return false;
         }
         if( verbose )
           std::cout << "Processing files " << inputFiles[ i ] << " and " << inputFiles[ i + 1 ] << "...           \r" << std::flush;
         if( ! v1.load( inputFiles[ i ] ) ||
             ! v2.load( inputFiles[ i + 1 ] ) )
         {
            std::cerr << "Unable to read the files " << inputFiles[ i ] << " and " << inputFiles[ i + 1 ] << "." << std::endl;
            outputFile.close();
            return false;
         }
         outputFile << std::setw( 6 ) << i/2 * snapshotPeriod << " ";
         i++;
      }
      if( mode == "sequence" )
      {
         if( i == 0 )
         {
            if( verbose )
              std::cout << "Reading the file " << inputFiles[ 0 ] << "...               \r" << std::flush;
            if( ! v1.load( inputFiles[ 0 ] ) )
            {
               std::cerr << "Unable to read the file " << inputFiles[ 0 ] << std::endl;
               outputFile.close();
               return false;
            }
         }
         if( verbose )
           std::cout << "Processing the files " << inputFiles[ 0 ] << " and " << inputFiles[ i ] << "...             \r" << std::flush;
         if( ! v2.load( inputFiles[ i ] ) )
         {
            std::cerr << "Unable to read the file " << inputFiles[ 1 ] << std::endl;
            outputFile.close();
            return false;
         }
         outputFile << std::setw( 6 ) << ( i - 1 ) * snapshotPeriod << " ";
      }
      if( mode == "halves" )
      {
         const int half = inputFiles. getSize() / 2;
         if( i == 0 )
            i = half;
         if( verbose )
           std::cout << "Processing files " << inputFiles[ i - half ] << " and " << inputFiles[ i ] << "...                 \r" << std::flush;
         if( ! v1.load( inputFiles[ i - half ] ) ||
             ! v2.load( inputFiles[ i ] ) )
         {
            std::cerr << "Unable to read the files " << inputFiles[ i - half ] << " and " << inputFiles[ i ] << "." << std::endl;
            outputFile.close();
            return false;
         }
         //if( snapshotPeriod != 0.0 )
         outputFile << std::setw( 6 ) << ( i - half ) * snapshotPeriod << " ";
      }
      diff = v1;
      diff -= v2;
      Real l1Diff = diff.getLpNorm( 1.0 );
      Real l2Diff = diff.getLpNorm( 2.0 );
      Real maxDiff = diff.getMaxNorm();
      if( snapshotPeriod != 0.0 )
      {
         totalL1Diff += snapshotPeriod * l1Diff;
         totalL2Diff += snapshotPeriod * l2Diff * l2Diff;
      }
      else
      {
         totalL1Diff += l1Diff;
         totalL2Diff += l2Diff * l2Diff;
      }
      totalMaxDiff = max( totalMaxDiff, maxDiff );
      outputFile << std::setw( 18 ) << l1Diff
                 << std::setw( 18 ) << l2Diff
                 << std::setw( 18 ) << maxDiff
                 << std::setw( 18 ) << totalL1Diff
                 << std::setw( 18 ) << ::sqrt( totalL2Diff )
                 << std::setw( 18 ) << totalMaxDiff << std::endl;

      if( writeDifference )
      {
         String differenceFileName;
         differenceFileName = inputFiles[ i ];
         removeFileExtension( differenceFileName );
         differenceFileName += ".diff.tnl";
         //diff.setLike( v1 );
         diff = v1;
         diff -= v2;
         diff.save( differenceFileName );
      }
   }
   outputFile.close();

   if( verbose )
     std::cout << std::endl;
   return true;
}


template< typename MeshPointer, typename Element, typename Real, typename Index >
bool computeDifferenceOfVectors( const MeshPointer& meshPointer, const Config::ParameterContainer& parameters )
{
   bool verbose = parameters. getParameter< bool >( "verbose" );
   Containers::List< String > inputFiles = parameters. getParameter< Containers::List< String > >( "input-files" );
   String mode = parameters. getParameter< String >( "mode" );
   String outputFileName = parameters. getParameter< String >( "output-file" );
   double snapshotPeriod = parameters. getParameter< double >( "snapshot-period" );
   bool writeDifference = parameters. getParameter< bool >( "write-difference" );

   std::fstream outputFile;
   outputFile.open( outputFileName.getString(), std::fstream::out );
   if( ! outputFile )
   {
      std::cerr << "Unable to open the file " << outputFileName << "." << std::endl;
      return false;
   }
   outputFile << "#";
   outputFile << std::setw( 6 ) << "Time";
   outputFile << std::setw( 18 ) << "L1 diff."
              << std::setw( 18 ) << "L2 diff."
              << std::setw( 18 ) << "Max. diff."
              << std::setw( 18 ) << "Total L1 diff."
              << std::setw( 18 ) << "Total L2 diff."
              << std::setw( 18 ) << "Total Max. diff."
              << std::endl;
   if( verbose )
     std::cout << std::endl;

   Containers::Vector< Real, Devices::Host, Index > v1, v2;
   Real totalL1Diff( 0.0 ), totalL2Diff( 0.0 ), totalMaxDiff( 0.0 );
   for( int i = 0; i < inputFiles. getSize(); i ++ )
   {
      if( mode == "couples" )
      {
         if( i + 1 == inputFiles.getSize() )
         {
            std::cerr << std::endl << "Skipping the file " << inputFiles[ i ] << " since there is no file to couple it with." << std::endl;
            outputFile.close();
            return false;
         }
         if( verbose )
           std::cout << "Processing files " << inputFiles[ i ] << " and " << inputFiles[ i + 1 ] << "...           \r" << std::flush;
         if( ! v1.load( inputFiles[ i ] ) ||
             ! v2.load( inputFiles[ i + 1 ] ) )
         {
            std::cerr << "Unable to read the files " << inputFiles[ i ] << " and " << inputFiles[ i + 1 ] << "." << std::endl;
            outputFile.close();
            return false;
         }
         outputFile << std::setw( 6 ) << i/2 * snapshotPeriod << " ";
         i++;
      }
      if( mode == "sequence" )
      {
         if( i == 0 )
         {
            if( verbose )
              std::cout << "Reading the file " << inputFiles[ 0 ] << "...               \r" << std::flush;
            if( ! v1.load( inputFiles[ 0 ] ) )
            {
               std::cerr << "Unable to read the file " << inputFiles[ 0 ] << std::endl;
               outputFile.close();
               return false;
            }
         }
         if( verbose )
           std::cout << "Processing the files " << inputFiles[ 0 ] << " and " << inputFiles[ i ] << "...             \r" << std::flush;
         if( ! v2.load( inputFiles[ i ] ) )
         {
            std::cerr << "Unable to read the file " << inputFiles[ 1 ] << std::endl;
            outputFile.close();
            return false;
         }
         outputFile << std::setw( 6 ) << ( i - 1 ) * snapshotPeriod << " ";
      }
      if( mode == "halves" )
      {
         const int half = inputFiles. getSize() / 2;
         if( i == 0 )
            i = half;
         if( verbose )
           std::cout << "Processing files " << inputFiles[ i - half ] << " and " << inputFiles[ i ] << "...                 \r" << std::flush;
         if( ! v1.load( inputFiles[ i - half ] ) ||
             ! v2.load( inputFiles[ i ] ) )
         {
            std::cerr << "Unable to read the files " << inputFiles[ i - half ] << " and " << inputFiles[ i ] << "." << std::endl;
            outputFile.close();
            return false;
         }
         //if( snapshotPeriod != 0.0 )
         outputFile << std::setw( 6 ) << ( i - half ) * snapshotPeriod << " ";
      }
      Real l1Diff = meshPointer->getDifferenceLpNorm( v1, v2, 1.0 );
      Real l2Diff = meshPointer->getDifferenceLpNorm( v1, v2, 2.0 );
      Real maxDiff = meshPointer->getDifferenceAbsMax( v1, v2 );
      if( snapshotPeriod != 0.0 )
      {
         totalL1Diff += snapshotPeriod * l1Diff;
         totalL2Diff += snapshotPeriod * l2Diff * l2Diff;
      }
      else
      {
         totalL1Diff += l1Diff;
         totalL2Diff += l2Diff * l2Diff;
      }
      totalMaxDiff = max( totalMaxDiff, maxDiff );
      outputFile << std::setw( 18 ) << l1Diff
                 << std::setw( 18 ) << l2Diff
                 << std::setw( 18 ) << maxDiff
                 << std::setw( 18 ) << totalL1Diff
                 << std::setw( 18 ) << ::sqrt( totalL2Diff )
                 << std::setw( 18 ) << totalMaxDiff << std::endl;

      if( writeDifference )
      {
         String differenceFileName;
         differenceFileName = inputFiles[ i ];
         removeFileExtension( differenceFileName );
         differenceFileName += ".diff.tnl";
         Containers::Vector< Real, Devices::Host, Index > diff;
         diff.setLike( v1 );
         diff = v1;
         diff -= v2;
         diff.save( differenceFileName );
      }
   }
   outputFile.close();

   if( verbose )
     std::cout << std::endl;
   return true;
}

template< typename MeshPointer, typename Element, typename Real, typename Index >
bool computeDifference( const MeshPointer& meshPointer, const String& objectType, const Config::ParameterContainer& parameters )
{
   if( objectType == "Functions::MeshFunction" ||
       objectType == "tnlMeshFunction" )  // TODO: remove deprecated type name
      return computeDifferenceOfMeshFunctions< MeshPointer, Element, Real, Index >( meshPointer, parameters );
   if( objectType == "Containers::Vector" ||
       objectType == "tnlVector" || objectType == "tnlSharedVector" )   // TODO: remove deprecated type name
      return computeDifferenceOfVectors< MeshPointer, Element, Real, Index >( meshPointer, parameters );
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
       parsedObjectType[ 0 ] == "tnlMultiVector" ||                       // TODO: remove deprecated type names
       parsedObjectType[ 0 ] == "tnlSharedMultiVector"   )                //
      indexType = parsedObjectType[ 4 ];
   if( parsedObjectType[ 0 ] == "Containers::Vector" ||
       parsedObjectType[ 0 ] == "tnlSharedVector" ||                     // TODO: remove deprecated type names
       parsedObjectType[ 0 ] == "tnlVector" )                            //
      indexType = parsedObjectType[ 3 ];

   if( parsedObjectType[ 0 ] == "Functions::MeshFunction" ||
       parsedObjectType[ 0 ] == "tnlMeshFunction" )                      // TODO: remove deprecated type names
      return computeDifference< MeshPointer, Element, Real, typename MeshPointer::ObjectType::IndexType >( meshPointer, parsedObjectType[ 0 ], parameters );
   
   if( indexType == "int" )
      return computeDifference< MeshPointer, Element, Real, int >( meshPointer, parsedObjectType[ 0 ], parameters );
   if( indexType == "long-int" )
      return computeDifference< MeshPointer, Element, Real, long int >( meshPointer, parsedObjectType[ 0 ], parameters );
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

   if( parsedObjectType[ 0 ] == "Containers::MultiVector" ||
       parsedObjectType[ 0 ] == "tnlMultiVector" ||                         // TODO: remove deprecated type names
       parsedObjectType[ 0 ] == "tnlSharedMultiVector" )                    //
      elementType = parsedObjectType[ 2 ];
   if( parsedObjectType[ 0 ] == "Functions::MeshFunction" ||
       parsedObjectType[ 0 ] == "tnlMeshFunction" )                         // TODO: remove deprecated type names
      elementType = parsedObjectType[ 3 ];
   if( parsedObjectType[ 0 ] == "Containers::Vector" ||
       parsedObjectType[ 0 ] == "tnlSharedVector" ||                        // TODO: remove deprecated type names
       parsedObjectType[ 0 ] == "tnlVector" )                               //
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
   if( parsedElementType[ 0 ] == "tnlStaticVector" )
      return setTupleType< MeshPointer >( meshPointer, inputFileName, parsedObjectType, parsedElementType, parameters );

   std::cerr << "Unknown element type " << elementType << "." << std::endl;
   return false;
}

template< typename Mesh >
bool processFiles( const Config::ParameterContainer& parameters )
{
   int verbose = parameters. getParameter< int >( "verbose");
   Containers::List< String > inputFiles = parameters. getParameter< Containers::List< String > >( "input-files" );
   String& inputFile = inputFiles[ 0 ];

   /****
    * Reading the mesh
    */
   String meshFile = parameters. getParameter< String >( "mesh" );
   
   typedef SharedPointer< Mesh > MeshPointer;

   MeshPointer meshPointer;
   if( meshFile != "" )
      if( ! meshPointer->load( meshFile ) )
      {
         std::cerr << "I am not able to load mesh from the file " << meshFile << "." << std::endl;
         return false;
      }

   String objectType;
   if( ! getObjectType( inputFiles[ 0 ], objectType ) ) {
       std::cerr << "unknown object ... SKIPPING!" << std::endl;
       return false;
   }

   if( verbose )
     std::cout << objectType << " detected ... ";

   Containers::List< String > parsedObjectType;
   if( ! parseObjectType( objectType, parsedObjectType ) )
   {
      std::cerr << "Unable to parse object type " << objectType << "." << std::endl;
      return false;
   }
   setElementType< MeshPointer >( meshPointer, inputFiles[ 0 ], parsedObjectType, parameters );
   return true;
}

#endif /* TNL_DIFF_H_ */
