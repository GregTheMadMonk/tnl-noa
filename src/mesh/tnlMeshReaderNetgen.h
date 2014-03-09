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

using namespace std;

class tnlMeshReaderNetgen
{
   public:

   static bool detectDimensions( const tnlString& fileName,
                                 int& dimensions )
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

      /****
       * Read the first vertex and compute number of components
       */
      if( ! inputFile )
         return false;
      getline( inputFile, line );
      iss.str( line );
      dimensions = -1;
      while( iss )
      {
         double aux;
         iss >> aux;
         dimensions++;
      }
      return true;
   }

   template< typename MeshType >
   static bool readMesh( const tnlString& fileName,
                         MeshType& mesh,
                         bool verbose )
   {
      typedef typename MeshType::PointType PointType;
      const int dimensions = PointType::size;

      fstream inputFile( fileName.getString() );
      if( ! inputFile )
      {
         cerr << "I am not able to open the file " << fileName << "." << endl;
         return false;
      }

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
      VertexIndexType numberOfVertices;
      iss >> numberOfVertices;
      if( ! mesh.setNumberOfVertices( numberOfVertices ) )
      {
         cerr << "I am not able to allocate enough memory for " << numberOfVertices << " vertices." << endl;
         return false;
      }

      for( VertexIndexType i = 0; i < numberOfVertices; i++ )
      {
         getline( inputFile, line );
         iss.clear();
         iss.str( line );
         PointType p;
         for( int d = 0; d < dimensions; d++ )
            iss >> p[ d ];
         mesh.setVertex( i, p );
         if( verbose )
            cout << numberOfVertices << " vertices expected ... " << i+1 << "/" << numberOfVertices << "        \r" << flush;
         const PointType& point = mesh.getVertex( i ).getPoint();
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
          return false;
       getline( inputFile, line );
       iss.str( line );
       CellIndexType numberOfCells;
       iss >> numberOfCells;
       if( ! mesh.template setNumberOfEntities< dimensions >( numberOfCells ) )
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
          for( int cellVertex = 0; cellVertex < dimensions + 1; cellVertex++ )
          {
             VertexIndexType vertexIdx;
             iss >> vertexIdx;
             mesh.template getEntity< dimensions >( i ).setVertexIndex( cellVertex, vertexIdx );
          }
          cout << endl;
          if( verbose )
             cout << numberOfCells << " cells expected ... " << i+1 << "/" << numberOfCells << "                 \r" << flush;
       }
       if( verbose )
          cout << endl;
       return true;
   }

   protected:


};


#endif /* TNLMESHREADERNETGEN_H_ */
