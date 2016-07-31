/***************************************************************************
                          tnl-grid-setup.h  -  description
                             -------------------
    begin                : Nov 20, 2013
    copyright            : (C) 2013 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#ifndef TNL_GRID_SETUP_H_
#define TNL_GRID_SETUP_H_

#include <TNL/Config/ParameterContainer.h>
#include <TNL/mesh/tnlGrid.h>

using namespace TNL;

template< typename RealType, typename IndexType >
bool setupGrid( const Config::ParameterContainer& parameters )
{
   String gridName = parameters. getParameter< String >( "grid-name" );
   String outputFile = parameters. getParameter< String >( "output-file" );
   int dimensions = parameters. getParameter< int >( "dimensions" );
   if( dimensions == 1 )
   {
      RealType originX = parameters. getParameter< double >( "origin-x" );
      RealType proportionsX = parameters. getParameter< double >( "proportions-x" );
      IndexType sizeX = parameters. getParameter< int >( "size-x" );
      std::cout << "Setting dimensions to  ... " << sizeX << std::endl;
      std::cout << "Writing the grid to the file " << outputFile << " .... ";

      typedef tnlGrid< 1, RealType, Devices::Host, IndexType > GridType;
      typedef typename GridType::VertexType VertexType;
      typedef typename GridType::CoordinatesType CoordinatesType;
      GridType grid;
      grid.setDomain( VertexType( originX ), VertexType( proportionsX ) );
      grid.setDimensions( CoordinatesType( sizeX ) );
      if( ! grid.save( outputFile ) )
      {
         std::cerr << "[ FAILED ] " << std::endl;
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
     std::cout << "Setting dimensions to  ... " << sizeX << "x" << sizeY << std::endl;
     std::cout << "Writing the grid to the file " << outputFile << " .... ";

      typedef tnlGrid< 2, RealType, Devices::Host, IndexType > GridType;
      typedef typename GridType::VertexType VertexType;
      typedef typename GridType::CoordinatesType CoordinatesType;
      GridType grid;
      grid.setDomain( VertexType( originX, originY ), VertexType( proportionsX, proportionsY ) );
      grid.setDimensions( CoordinatesType( sizeX, sizeY ) );
      if( ! grid.save( outputFile ) )
      {
         std::cerr << "[ FAILED ] " << std::endl;
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
     std::cout << "Setting dimensions to  ... " << sizeX << "x" << sizeY << "x" << sizeZ << std::endl;
     std::cout << "Writing the grid to the file " << outputFile << " .... ";

      typedef tnlGrid< 3, RealType, Devices::Host, IndexType > GridType;
      typedef typename GridType::VertexType VertexType;
      typedef typename GridType::CoordinatesType CoordinatesType;
      GridType grid;
      grid.setDomain( VertexType( originX, originY, originZ ), VertexType( proportionsX, proportionsY, proportionsZ ) );
      grid.setDimensions( CoordinatesType( sizeX, sizeY, sizeZ ) );
      if( ! grid.save( outputFile ) )
      {
         std::cerr << "[ FAILED ] " << std::endl;
         return false;
      }
   }
  std::cout << "[ OK ] " << std::endl;
   return true;
}

template< typename RealType >
bool resolveIndexType( const Config::ParameterContainer& parameters )
{
   const String& indexType = parameters. getParameter< String >( "index-type" );
  std::cout << "Setting index type to  ... " << indexType << std::endl;
   if( indexType == "int" )
      return setupGrid< RealType, int >( parameters );
   if( indexType == "long int" )
      return setupGrid< RealType, long int >( parameters );
   std::cerr << "The index type '" << indexType << "' is not defined. " << std::endl;
   return false;
}

bool resolveRealType( const Config::ParameterContainer& parameters )
{
   String realType = parameters. getParameter< String >( "real-type" );
  std::cout << "Setting real type to   ... " << realType << std::endl;
   if( realType == "float" )
      return resolveIndexType< float >( parameters );
   if( realType == "double" )
      return resolveIndexType< double >( parameters );
   if( realType == "long-double" )
      return resolveIndexType< long double >( parameters );
   std::cerr << "The real type '" << realType << "' is not supported. " << std::endl;
   return false;
}

#endif /* TNL_GRID_SETUP_H_ */
