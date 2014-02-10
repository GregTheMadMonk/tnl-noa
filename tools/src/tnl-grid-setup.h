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
   tnlString gridName = parameters. GetParameter< tnlString >( "grid-name" );
   tnlString outputFile = parameters. GetParameter< tnlString >( "output-file" );
   int dimensions = parameters. GetParameter< int >( "dimensions" );
   cout << "Writing the grid to the file " << outputFile << " .... ";
   if( dimensions == 1 )
   {
      RealType originX = parameters. GetParameter< double >( "origin-x" );
      RealType proportionsX = parameters. GetParameter< double >( "proportions-x" );
      IndexType sizeX = parameters. GetParameter< int >( "size-x" );

      tnlGrid< 1, RealType, tnlHost, IndexType > grid;
      grid.setName( gridName );
      grid.setOrigin( tnlStaticVector< 1, RealType >( originX ) );
      grid.setProportions( tnlStaticVector< 1, RealType >( proportionsX ) );
      grid.setDimensions( tnlStaticVector< 1, IndexType >( sizeX ) );
      if( ! grid.save( outputFile ) )
      {
         cerr << "[ FAILED ] " << endl;
         return false;
      }
   }
   if( dimensions == 2 )
   {
      RealType originX = parameters. GetParameter< double >( "origin-x" );
      RealType originY = parameters. GetParameter< double >( "origin-y" );
      RealType proportionsX = parameters. GetParameter< double >( "proportions-x" );
      RealType proportionsY = parameters. GetParameter< double >( "proportions-y" );
      IndexType sizeX = parameters. GetParameter< int >( "size-x" );
      IndexType sizeY = parameters. GetParameter< int >( "size-y" );

      tnlGrid< 2, RealType, tnlHost, IndexType > grid;
      grid.setName( gridName );
      grid.setOrigin( tnlStaticVector< 2, RealType >( originX, originY ) );
      grid.setProportions( tnlStaticVector< 2, RealType >( proportionsX, proportionsY ) );
      grid.setDimensions( tnlStaticVector< 2, IndexType >( sizeX, sizeY ) );
      if( ! grid.save( outputFile ) )
      {
         cerr << "[ FAILED ] " << endl;
         return false;
      }
   }
   if( dimensions == 3 )
   {
      RealType originX = parameters. GetParameter< double >( "origin-x" );
      RealType originY = parameters. GetParameter< double >( "origin-y" );
      RealType originZ = parameters. GetParameter< double >( "origin-z" );
      RealType proportionsX = parameters. GetParameter< double >( "proportions-x" );
      RealType proportionsY = parameters. GetParameter< double >( "proportions-y" );
      RealType proportionsZ = parameters. GetParameter< double >( "proportions-z" );
      IndexType sizeX = parameters. GetParameter< int >( "size-x" );
      IndexType sizeY = parameters. GetParameter< int >( "size-y" );
      IndexType sizeZ = parameters. GetParameter< int >( "size-z" );

      tnlGrid< 3, RealType, tnlHost, IndexType > grid;
      grid.setName( gridName );
      grid.setOrigin( tnlStaticVector< 3, RealType >( originX, originY, originZ ) );
      grid.setProportions( tnlStaticVector< 3, RealType >( proportionsX, proportionsY, proportionsZ ) );
      grid.setDimensions( tnlStaticVector< 3, IndexType >( sizeX, sizeY, sizeZ ) );
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
   const tnlString& indexType = parameters. GetParameter< tnlString >( "index-type" );
   cout << "Setting  to  ... " << indexType << endl;
   if( indexType == "int" )
      return setupGrid< RealType, int >( parameters );
   if( indexType == "long int" )
      return setupGrid< RealType, long int >( parameters );
   cerr << "The index type '" << indexType << "' is not defined. " << endl;
   return false;
}

bool resolveRealType( const tnlParameterContainer& parameters )
{
   tnlString realType = parameters. GetParameter< tnlString >( "real-type" );
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
