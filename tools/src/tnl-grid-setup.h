/***************************************************************************
                          tnl-grid-setup.h  -  description
                             -------------------
    begin                : Nov 20, 2013
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

#ifndef TNL_GRID_SETUP_H_
#define TNL_GRID_SETUP_H_

#include <config/tnlParameterContainer.h>
#include <mesh/tnlGrid.h>



template< typename RealType, typename IndexType >
bool setupGrid( const tnlParameterContainer& parameters )
{
   tnlString gridName = parameters. getParameter< tnlString >( "grid-name" );
   tnlString outputFile = parameters. getParameter< tnlString >( "output-file" );
   int dimensions = parameters. getParameter< int >( "dimensions" );
   if( dimensions == 1 )
   {
      RealType originX = parameters. getParameter< double >( "origin-x" );
      RealType proportionsX = parameters. getParameter< double >( "proportions-x" );
      IndexType sizeX = parameters. getParameter< int >( "size-x" );
      cout << "Setting dimensions to  ... " << sizeX << endl;
      cout << "Writing the grid to the file " << outputFile << " .... ";

      typedef tnlGrid< 1, RealType, tnlHost, IndexType > GridType;
      typedef typename GridType::VertexType VertexType;
      typedef typename GridType::CoordinatesType CoordinatesType;
      GridType grid;
      grid.setName( gridName );
      grid.setDomain( VertexType( originX ), VertexType( proportionsX ) );
      grid.setDimensions( CoordinatesType( sizeX ) );
      if( ! grid.save( outputFile ) )
      {
         cerr << "[ FAILED ] " << endl;
         return false;
      }
   }
   if( dimensions == 2 )
   {
      RealType originX = parameters. getParameter< double >( "origin-x" );
      RealType originY = parameters. getParameter< double >( "origin-y" );
      RealType proportionsX = parameters. getParameter< double >( "proportions-x" );
      RealType proportionsY = parameters. getParameter< double >( "proportions-y" );
      IndexType sizeX = parameters. getParameter< int >( "size-x" );
      IndexType sizeY = parameters. getParameter< int >( "size-y" );
      cout << "Setting dimensions to  ... " << sizeX << "x" << sizeY << endl;
      cout << "Writing the grid to the file " << outputFile << " .... ";

      typedef tnlGrid< 2, RealType, tnlHost, IndexType > GridType;
      typedef typename GridType::VertexType VertexType;
      typedef typename GridType::CoordinatesType CoordinatesType;
      GridType grid;
      grid.setName( gridName );
      grid.setDomain( VertexType( originX, originY ), VertexType( proportionsX, proportionsY ) );
      grid.setDimensions( CoordinatesType( sizeX, sizeY ) );
      if( ! grid.save( outputFile ) )
      {
         cerr << "[ FAILED ] " << endl;
         return false;
      }
   }
   if( dimensions == 3 )
   {
      RealType originX = parameters. getParameter< double >( "origin-x" );
      RealType originY = parameters. getParameter< double >( "origin-y" );
      RealType originZ = parameters. getParameter< double >( "origin-z" );
      RealType proportionsX = parameters. getParameter< double >( "proportions-x" );
      RealType proportionsY = parameters. getParameter< double >( "proportions-y" );
      RealType proportionsZ = parameters. getParameter< double >( "proportions-z" );
      IndexType sizeX = parameters. getParameter< int >( "size-x" );
      IndexType sizeY = parameters. getParameter< int >( "size-y" );
      IndexType sizeZ = parameters. getParameter< int >( "size-z" );
      cout << "Setting dimensions to  ... " << sizeX << "x" << sizeY << "x" << sizeZ << endl;
      cout << "Writing the grid to the file " << outputFile << " .... ";

      typedef tnlGrid< 3, RealType, tnlHost, IndexType > GridType;
      typedef typename GridType::VertexType VertexType;
      typedef typename GridType::CoordinatesType CoordinatesType;
      GridType grid;
      grid.setName( gridName );
      grid.setDomain( VertexType( originX, originY, originZ ), VertexType( proportionsX, proportionsY, proportionsZ ) );
      grid.setDimensions( CoordinatesType( sizeX, sizeY, sizeZ ) );
      if( ! grid.save( outputFile ) )
      {
         cerr << "[ FAILED ] " << endl;
         return false;
      }
   }
   cout << "[ OK ] " << endl;
   return true;
}

template< typename RealType >
bool resolveIndexType( const tnlParameterContainer& parameters )
{
   const tnlString& indexType = parameters. getParameter< tnlString >( "index-type" );
   cout << "Setting index type to  ... " << indexType << endl;
   if( indexType == "int" )
      return setupGrid< RealType, int >( parameters );
   if( indexType == "long int" )
      return setupGrid< RealType, long int >( parameters );
   cerr << "The index type '" << indexType << "' is not defined. " << endl;
   return false;
}

bool resolveRealType( const tnlParameterContainer& parameters )
{
   tnlString realType = parameters. getParameter< tnlString >( "real-type" );
   cout << "Setting real type to   ... " << realType << endl;
   if( realType == "float" )
      return resolveIndexType< float >( parameters );
   if( realType == "double" )
      return resolveIndexType< double >( parameters );
   if( realType == "long-double" )
      return resolveIndexType< long double >( parameters );
   cerr << "The real type '" << realType << "' is not supported. " << endl;
   return false;
}

#endif /* TNL_GRID_SETUP_H_ */
