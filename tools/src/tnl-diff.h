/***************************************************************************
                          tnl-diff.h  -  description
                             -------------------
    begin                : Nov 17, 2013
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

#ifndef TNL_DIFF_H_
#define TNL_DIFF_H_

#include <iomanip>
#include <config/tnlParameterContainer.h>
#include <core/mfilename.h>
#include <core/vectors/tnlVector.h>
#include <core/vectors/tnlStaticVector.h>

template< typename Mesh, typename Element, typename Real, typename Index >
bool computeDifference( const Mesh& mesh, const tnlParameterContainer& parameters )
{
   bool verbose = parameters. GetParameter< bool >( "verbose" );
   tnlList< tnlString > inputFiles = parameters. GetParameter< tnlList< tnlString > >( "input-files" );
   tnlString mode = parameters. GetParameter< tnlString >( "mode" );
   tnlString outputFileName = parameters. GetParameter< tnlString >( "output-file" );
   double snapshotPeriod = parameters. GetParameter< double >( "snapshot-period" );
   bool writeDifference = parameters. GetParameter< bool >( "write-difference" );

   fstream outputFile;
   outputFile.open( outputFileName.getString(), std::fstream::out );
   if( ! outputFile )
   {
      cerr << "Unable to open the file " << outputFileName << "." << endl;
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
              << endl;
   if( verbose )
      cout << endl;

   tnlVector< Real, tnlHost, Index > v1, v2;
   Real totalL1Diff( 0.0 ), totalL2Diff( 0.0 ), totalMaxDiff( 0.0 );
   for( int i = 0; i < inputFiles. getSize(); i ++ )
   {
      if( mode == "couples" )
      {
         if( i + 1 == inputFiles.getSize() )
         {
            cerr << endl << "Skipping the file " << inputFiles[ i ] << " since there is no file to couple it with." << endl;
            outputFile.close();
            return false;
         }
         if( verbose )
            cout << "Processing files " << inputFiles[ i ] << " and " << inputFiles[ i + 1 ] << "...           \r" << flush;
         if( ! v1.load( inputFiles[ i ] ) ||
             ! v2.load( inputFiles[ i + 1 ] ) )
         {
            cerr << "Unable to read the files " << inputFiles[ i ] << " and " << inputFiles[ i + 1 ] << "." << endl;
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
               cout << "Reading the file " << inputFiles[ 0 ] << "...               \r" << flush;
            if( ! v1.load( inputFiles[ 0 ] ) )
            {
               cerr << "Unable to read the file " << inputFiles[ 0 ] << endl;
               outputFile.close();
               return false;
            }
         }
         if( verbose )
            cout << "Processing the files " << inputFiles[ 0 ] << " and " << inputFiles[ i ] << "...             \r" << flush;
         if( ! v2.load( inputFiles[ i ] ) )
         {
            cerr << "Unable to read the file " << inputFiles[ 1 ] << endl;
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
            cout << "Processing files " << inputFiles[ i - half ] << " and " << inputFiles[ i ] << "...                 \r" << flush;
         if( ! v1.load( inputFiles[ i - half ] ) ||
             ! v2.load( inputFiles[ i ] ) )
         {
            cerr << "Unable to read the files " << inputFiles[ i - half ] << " and " << inputFiles[ i ] << "." << endl;
            outputFile.close();
            return false;
         }
         //if( snapshotPeriod != 0.0 )
         outputFile << std::setw( 6 ) << ( i - half ) * snapshotPeriod << " ";
      }
      Real l1Diff = mesh.getDifferenceLpNorm( v1, v2, 1.0 );
      Real l2Diff = mesh.getDifferenceLpNorm( v1, v2, 2.0 );
      Real maxDiff = mesh.getDifferenceAbsMax( v1, v2 );
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
      totalMaxDiff = Max( totalMaxDiff, maxDiff );
      outputFile << std::setw( 18 ) << l1Diff
                 << std::setw( 18 ) << l2Diff
                 << std::setw( 18 ) << maxDiff
                 << std::setw( 18 ) << totalL1Diff
                 << std::setw( 18 ) << sqrt( totalL2Diff )
                 << std::setw( 18 ) << totalMaxDiff << endl;

      if( writeDifference )
      {
         tnlString differenceFileName;
         differenceFileName = inputFiles[ i ];
         RemoveFileExtension( differenceFileName );
         differenceFileName += ".diff.tnl";
         tnlVector< Real, tnlHost, Index > diff;
         diff.setLike( v1 );
         diff = v1;
         diff -= v2;
         diff.save( differenceFileName );
      }
   }
   outputFile.close();

   if( verbose )
      cout << endl;
   return true;
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
      return computeDifference< Mesh, Element, Real, int >( mesh, parameters );
   if( indexType == "long-int" )
      return computeDifference< Mesh, Element, Real, long int >( mesh, parameters );
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
            return setIndexType< Mesh, tnlStaticVector< 1, float >, float >( mesh, inputFileName, parsedObjectType, parameters );
            break;
         case 2:
            return setIndexType< Mesh, tnlStaticVector< 2, float >, float >( mesh, inputFileName, parsedObjectType, parameters );
            break;
         case 3:
            return setIndexType< Mesh, tnlStaticVector< 3, float >, float >( mesh, inputFileName, parsedObjectType, parameters );
            break;
      }
   if( dataType == "double" )
      switch( dimensions )
      {
         case 1:
            return setIndexType< Mesh, tnlStaticVector< 1, double >, double >( mesh, inputFileName, parsedObjectType, parameters );
            break;
         case 2:
            return setIndexType< Mesh, tnlStaticVector< 2, double >, double >( mesh, inputFileName, parsedObjectType, parameters );
            break;
         case 3:
            return setIndexType< Mesh, tnlStaticVector< 3, double >, double >( mesh, inputFileName, parsedObjectType, parameters );
            break;
      }
   if( dataType == "long double" )
      switch( dimensions )
      {
         case 1:
            return setIndexType< Mesh, tnlStaticVector< 1, long double >, long double >( mesh, inputFileName, parsedObjectType, parameters );
            break;
         case 2:
            return setIndexType< Mesh, tnlStaticVector< 2, long double >, long double >( mesh, inputFileName, parsedObjectType, parameters );
            break;
         case 3:
            return setIndexType< Mesh, tnlStaticVector< 3, long double >, long double >( mesh, inputFileName, parsedObjectType, parameters );
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
   if( parsedElementType[ 0 ] == "tnlStaticVector" )
      return setTupleType< Mesh >( mesh, inputFileName, parsedObjectType, parsedElementType, parameters );

   cerr << "Unknown element type " << elementType << "." << endl;
   return false;
}

template< typename Mesh >
bool processFiles( const tnlParameterContainer& parameters )
{
   int verbose = parameters. GetParameter< int >( "verbose");
   tnlList< tnlString > inputFiles = parameters. GetParameter< tnlList< tnlString > >( "input-files" );
   tnlString& inputFile = inputFiles[ 0 ];

   /****
    * Reading the mesh
    */
   tnlString meshFile = parameters. GetParameter< tnlString >( "mesh" );

   Mesh mesh;
   if( meshFile != "" )
      if( ! mesh. load( meshFile ) )
      {
         cerr << "I am not able to load mesh from the file " << meshFile << "." << endl;
         return false;
      }

   tnlString objectType;
   if( ! getObjectType( inputFiles[ 0 ], objectType ) )
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
      setElementType< Mesh >( mesh, inputFiles[ 0 ], parsedObjectType, parameters );
   }
}

#endif /* TNL_DIFF_H_ */
