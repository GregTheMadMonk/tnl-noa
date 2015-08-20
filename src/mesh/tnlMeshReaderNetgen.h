/***************************************************************************
                          tnlMeshReaderNetgen.h  -  description
                             -------------------
    begin                : Feb 19, 2014
    copyright            : (C) 2014 by Tomas Oberhuber
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

#ifndef TNLMESHREADERNETGEN_H_
#define TNLMESHREADERNETGEN_H_

#include <fstream>
#include <istream>
#include <sstream>

#include <mesh/tnlMeshBuilder.h>

using namespace std;

class tnlMeshReaderNetgen
{
   public:

      tnlMeshReaderNetgen()
      : dimensions( 0 ){}
      
   bool detectMesh( const tnlString& fileName )
   {
      fstream inputFile( fileName.getString() );
      if( ! inputFile )
      {
         cerr << "I am not able to open the file " << fileName << "." << endl;
         return false;
      }

      string line;
      istringstream iss;

      /****
       * Skip whitespaces
       */
      inputFile >> ws;
      
      /****
       * Skip number of vertices
       */
      if( ! inputFile )
         return false;
      getline( inputFile, line );
      iss.str( line );
      long int numberOfVertices;
      iss >> numberOfVertices;
      
      //cout << "There are " << numberOfVertices << " vertices." << endl;

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
            cerr << "The mesh file " << fileName << " is probably corrupted, some vertices are missing." << endl;
            return false;
         }
         verticesRead++;
      }
      
      /****
       * Skip whitespaces
       */
      inputFile >> ws;
         
      /****
       * Get number of cells
       */
      long int numberOfCells;
      getline( inputFile, line );
      iss.clear();
      iss.str( line );
      iss >> numberOfCells;
      //cout << "There are " << numberOfCells << " cells." << endl;
      
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
      //cout << "There are " << this->verticesInCell << " vertices in cell ..." << endl;
      return true;
   }

   template< typename MeshType >
   static bool readMesh( const tnlString& fileName,
                         MeshType& mesh,
                         bool verbose )
   {
      typedef typename MeshType::PointType PointType;
      typedef tnlMeshBuilder< MeshType > MeshBuilder;
      
      const int dimensions = PointType::size;

      fstream inputFile( fileName.getString() );
      if( ! inputFile )
      {
         cerr << "I am not able to open the file " << fileName << "." << endl;
         return false;
      }

      MeshBuilder meshBuilder;
      string line;
      istringstream iss;

      /****
       * Skip white spaces
       */
      inputFile >> ws;

      /****
       * Read the number of vertices
       */
      if( ! inputFile )
         return false;
      getline( inputFile, line );
      iss.str( line );
      typedef typename MeshType::template EntitiesTraits< 0 >::GlobalIndexType VertexIndexType;
      VertexIndexType pointsCount;
      iss >> pointsCount;
      if( ! meshBuilder.setPointsCount( pointsCount ) )
      {
         cerr << "I am not able to allocate enough memory for " << pointsCount << " vertices." << endl;
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
         cout << "Setting point number " << i << " of " << pointsCount << endl;
         meshBuilder.setPoint( i, p );
         if( verbose )
            cout << pointsCount << " vertices expected ... " << i+1 << "/" << pointsCount << "        \r" << flush;
         //const PointType& point = mesh.getVertex( i ).getPoint();
      }
      if( verbose )
         cout << endl;

      /****
        * Skip white spaces
        */
       inputFile >> ws;

      /****
       * Read number of cells
       */
       typedef typename MeshType::template EntitiesTraits< dimensions >::GlobalIndexType CellIndexType;
       if( ! inputFile )
       {
          cerr << "I cannot read the mesh cells." << endl;
          return false;
       }
       getline( inputFile, line );
       iss.clear();
       iss.str( line );
       CellIndexType numberOfCells=atoi( line.data() );
       //iss >> numberOfCells; // TODO: I do not know why this does not work
       if( ! meshBuilder.setCellsCount( numberOfCells ) )
       {
          cerr << "I am not able to allocate enough memory for " << numberOfCells << " cells." << endl;
          return false;
       }
       for( CellIndexType i = 0; i < numberOfCells; i++ )
       {
          getline( inputFile, line );
          iss.clear();
          iss.str( line );
          int subdomainIndex;
          iss >> subdomainIndex;
          cout << "Setting cell number " << i << " of " << numberOfCells << endl;
          typedef typename MeshBuilder::CellSeedType CellSeedType;
          for( int cellVertex = 0; cellVertex < CellSeedType::getCornersCount(); cellVertex++ )
          {
             VertexIndexType vertexIdx;
             iss >> vertexIdx;
             meshBuilder.getCellSeed( i ).setCornerId( cellVertex, vertexIdx - 1 );
          }
          if( verbose )
             cout << numberOfCells << " cells expected ... " << i+1 << "/" << numberOfCells << "                 \r" << flush;
       }
       if( verbose )
          cout << endl;
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


#endif /* TNLMESHREADERNETGEN_H_ */
