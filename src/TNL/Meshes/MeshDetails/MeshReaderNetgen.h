/***************************************************************************
                          MeshReaderNetgen.h  -  description
                             -------------------
    begin                : Feb 19, 2014
    copyright            : (C) 2014 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <fstream>
#include <istream>
#include <sstream>

#include <TNL/Meshes/MeshBuilder.h>

namespace TNL {
namespace Meshes {

class MeshReaderNetgen
{
   public:

      MeshReaderNetgen()
      : dimensions( 0 ){}
 
   bool detectMesh( const String& fileName )
   {
      std::fstream inputFile( fileName.getString() );
      if( ! inputFile )
      {
         std::cerr << "I am not able to open the file " << fileName << "." << std::endl;
         return false;
      }

      std::string line;
      std::istringstream iss;

      /****
       * Skip whitespaces
       */
      inputFile >> std::ws;
 
      /****
       * Skip number of vertices
       */
      if( ! inputFile )
         return false;
      getline( inputFile, line );
      iss.str( line );
      long int numberOfVertices;
      iss >> numberOfVertices;
 
      //cout << "There are " << numberOfVertices << " vertices." << std::endl;

      /****
       * Read the first vertex and compute number of components
       */
      if( ! inputFile )
         return false;
      getline( inputFile, line );
      iss.clear();
      iss.str( line );
      this->dimensions = -1;
      while( iss )
      {
         double aux;
         iss >> aux;
         this->dimensions++;
      }
 
      /****
       * Skip vertices
       */
      long int verticesRead( 1 );
      while( verticesRead < numberOfVertices )
      {
         getline( inputFile, line );
         if( ! inputFile )
         {
            std::cerr << "The mesh file " << fileName << " is probably corrupted, some vertices are missing." << std::endl;
            return false;
         }
         verticesRead++;
      }
 
      /****
       * Skip whitespaces
       */
      inputFile >> std::ws;
 
      /****
       * Get number of cells
       */
      long int numberOfCells;
      getline( inputFile, line );
      iss.clear();
      iss.str( line );
      iss >> numberOfCells;
      //cout << "There are " << numberOfCells << " cells." << std::endl;
 
      /****
       * Get number of vertices in a cell
       */
      getline( inputFile, line );
      iss.clear();
      iss.str( line );
      this->verticesInCell = -2;
      while( iss )
      {
         int aux;
         iss >> aux;
         this->verticesInCell++;
      }
      //cout << "There are " << this->verticesInCell << " vertices in cell ..." << std::endl;
      return true;
   }

   template< typename MeshType >
   static bool readMesh( const String& fileName,
                         MeshType& mesh,
                         bool verbose )
   {
      typedef typename MeshType::PointType PointType;
      typedef MeshBuilder< MeshType > MeshBuilder;
 
      const int dimensions = PointType::size;

      std::fstream inputFile( fileName.getString() );
      if( ! inputFile )
      {
         std::cerr << "I am not able to open the file " << fileName << "." << std::endl;
         return false;
      }

      MeshBuilder meshBuilder;
      std::string line;
      std::istringstream iss;

      /****
       * Skip white spaces
       */
      inputFile >> std::ws;

      /****
       * Read the number of vertices
       */
      if( ! inputFile )
         return false;
      getline( inputFile, line );
      iss.str( line );
      typedef typename MeshType::MeshTraitsType::template EntityTraits< 0 >::GlobalIndexType VertexIndexType;
      VertexIndexType pointsCount;
      iss >> pointsCount;
      if( ! meshBuilder.setPointsCount( pointsCount ) )
      {
         std::cerr << "I am not able to allocate enough memory for " << pointsCount << " vertices." << std::endl;
         return false;
      }

      for( VertexIndexType i = 0; i < pointsCount; i++ )
      {
         getline( inputFile, line );
         iss.clear();
         iss.str( line );
         PointType p;
         for( int d = 0; d < dimensions; d++ )
            iss >> p[ d ];
         //cout << "Setting point number " << i << " of " << pointsCount << std::endl;
         meshBuilder.setPoint( i, p );
         if( verbose )
           std::cout << pointsCount << " vertices expected ... " << i+1 << "/" << pointsCount << "        \r" << std::flush;
         //const PointType& point = mesh.getVertex( i ).getPoint();
      }
      if( verbose )
        std::cout << std::endl;

      /****
        * Skip white spaces
        */
       inputFile >> std::ws;

      /****
       * Read number of cells
       */
       typedef typename MeshType::MeshTraitsType::template EntityTraits< dimensions >::GlobalIndexType CellIndexType;
       if( ! inputFile )
       {
          std::cerr << "I cannot read the mesh cells." << std::endl;
          return false;
       }
       getline( inputFile, line );
       iss.clear();
       iss.str( line );
       CellIndexType numberOfCells=atoi( line.data() );
       //iss >> numberOfCells; // TODO: I do not know why this does not work
       if( ! meshBuilder.setCellsCount( numberOfCells ) )
       {
          std::cerr << "I am not able to allocate enough memory for " << numberOfCells << " cells." << std::endl;
          return false;
       }
       for( CellIndexType i = 0; i < numberOfCells; i++ )
       {
          getline( inputFile, line );
          iss.clear();
          iss.str( line );
          int subdomainIndex;
          iss >> subdomainIndex;
          //cout << "Setting cell number " << i << " of " << numberOfCells << std::endl;
          typedef typename MeshBuilder::CellSeedType CellSeedType;
          for( int cellVertex = 0; cellVertex < CellSeedType::getCornersCount(); cellVertex++ )
          {
             VertexIndexType vertexIdx;
             iss >> vertexIdx;
             meshBuilder.getCellSeed( i ).setCornerId( cellVertex, vertexIdx - 1 );
          }
          if( verbose )
            std::cout << numberOfCells << " cells expected ... " << i+1 << "/" << numberOfCells << "                 \r" << std::flush;
       }
       if( verbose )
         std::cout << std::endl;
       meshBuilder.build( mesh );
       return true;
   }

   int getDimensions() const
   {
      return this->dimensions;
   }
 
   int getVerticesInCell() const
   {
      return this->verticesInCell;
   }
 
   protected:

      int dimensions, verticesInCell;

};

} // namespace Meshes
} // namespace TNL
