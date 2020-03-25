/***************************************************************************
                          NetgenReader.h  -  description
                             -------------------
    begin                : Feb 19, 2014
    copyright            : (C) 2014 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

/***
 * Authors:
 * Oberhuber Tomas, tomas.oberhuber@fjfi.cvut.cz
 * Zabka Vitezslav, zabkav@gmail.com
 */

#pragma once

#include <fstream>
#include <istream>
#include <sstream>

#include <TNL/Meshes/MeshBuilder.h>
#include <TNL/Meshes/VTKTraits.h>

namespace TNL {
namespace Meshes {
namespace Readers {

class NetgenReader
{
public:
   bool detectMesh( const String& fileName )
   {
      this->reset();
      this->fileName = fileName;

      std::ifstream inputFile( fileName.getString() );
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
      meshDimension = worldDimension = -1;
      while( iss )
      {
         double aux;
         iss >> aux;
         meshDimension = ++worldDimension;
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
      int verticesInCell = -2;
      while( iss )
      {
         int aux;
         iss >> aux;
         verticesInCell++;
      }
      //cout << "There are " << verticesInCell << " vertices in cell ..." << std::endl;

      if( meshDimension == 1 && verticesInCell == 2 )
         cellShape = VTK::EntityShape::Line;
      else if( meshDimension == 2 ) {
         if( verticesInCell == 3 )
            cellShape = VTK::EntityShape::Triangle;
         else if( verticesInCell == 4 )
            cellShape = VTK::EntityShape::Quad;
      }
      else if( meshDimension == 3 ) {
         if( verticesInCell == 4 )
            cellShape = VTK::EntityShape::Tetra;
         else if( verticesInCell == 8 )
            cellShape = VTK::EntityShape::Hexahedron;
      }
      if( cellShape == VTK::EntityShape::Vertex ) {
         std::cerr << "Unknown cell topology: mesh dimension is " << meshDimension << ", number of vertices in cells is " << verticesInCell << "." << std::endl;
         return false;
      }

      return true;
   }

   template< typename MeshType >
   static bool readMesh( const String& fileName, MeshType& mesh )
   {
      typedef typename MeshType::PointType PointType;
      typedef MeshBuilder< MeshType > MeshBuilder;

      const int dimension = PointType::getSize();

      std::ifstream inputFile( fileName.getString() );
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
      typedef typename MeshType::Config::GlobalIndexType VertexIndexType;
      VertexIndexType pointsCount;
      iss >> pointsCount;
      meshBuilder.setPointsCount( pointsCount );

      for( VertexIndexType i = 0; i < pointsCount; i++ )
      {
         getline( inputFile, line );
         iss.clear();
         iss.str( line );
         PointType p;
         for( int d = 0; d < dimension; d++ )
            iss >> p[ d ];
         //cout << "Setting point number " << i << " of " << pointsCount << std::endl;
         meshBuilder.setPoint( i, p );
         //const PointType& point = mesh.getVertex( i ).getPoint();
      }

      /****
        * Skip white spaces
        */
      inputFile >> std::ws;

      /****
       * Read number of cells
       */
      typedef typename MeshType::Config::GlobalIndexType CellIndexType;
      if( ! inputFile )
      {
         std::cerr << "I cannot read the mesh cells." << std::endl;
         return false;
      }
      getline( inputFile, line );
      iss.clear();
      iss.str( line );
      CellIndexType numberOfCells = atoi( line.data() );
      //iss >> numberOfCells; // TODO: I do not know why this does not work
      meshBuilder.setCellsCount( numberOfCells );
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
      }
      meshBuilder.build( mesh );
      return true;
   }

   String
   getMeshType() const
   {
      return "Meshes::Mesh";
   }

   int getMeshDimension() const
   {
      return this->meshDimension;
   }
 
   int
   getWorldDimension() const
   {
      return worldDimension;
   }

   VTK::EntityShape
   getCellShape() const
   {
      return cellShape;
   }

   String
   getRealType() const
   {
      // not stored in the Netgen file
      return "float";
   }

   String
   getGlobalIndexType() const
   {
      // not stored in the Netgen file
      return "int";
   }
 
   String
   getLocalIndexType() const
   {
      // not stored in the Netgen file
      return "short int";
   }
 
protected:
   String fileName;
   int meshDimension, worldDimension;
   VTK::EntityShape cellShape = VTK::EntityShape::Vertex;

   void reset()
   {
      fileName = "";
      meshDimension = worldDimension = 0;
      cellShape = VTK::EntityShape::Vertex;
   }
};

} // namespace Readers
} // namespace Meshes
} // namespace TNL
