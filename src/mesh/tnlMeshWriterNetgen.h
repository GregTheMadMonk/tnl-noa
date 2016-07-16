/***************************************************************************
                          tnlMeshWriterNetgen.h  -  description
                             -------------------
    begin                : Feb 22, 2014
    copyright            : (C) 2014 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#ifndef TNLMESHWRITERNETGEN_H_
#define TNLMESHWRITERNETGEN_H_

#include <fstream>
#include <istream>
#include <sstream>
#include <iomanip>

using namespace std;

class tnlMeshWriterNetgen
{
   public:

   template< typename MeshType >
   static bool writeMesh( const tnlString& fileName,
                          MeshType& mesh,
                          bool verbose )
   {
      fstream outputFile;
      outputFile.open( fileName.getString(), ios::out );
      if( ! outputFile )
      {
         cerr << "I am not able to open the output file " << fileName << "." << endl;
         return false;
      }
      outputFile << setprecision( 6 );
      outputFile << fixed;

      const int meshDimensions = MeshType::meshDimensions;
      typedef typename MeshType::template EntitiesTraits< 0 >::GlobalIndexType VerticesIndexType;
      typedef typename MeshType::PointType                                     PointType;
      const VerticesIndexType numberOfVertices = mesh.getNumberOfVertices();
      outputFile << numberOfVertices << endl;
      for( VerticesIndexType i = 0; i < numberOfVertices; i++ )
      {
         const PointType& point = mesh.getVertex( i ).getPoint();
         outputFile << " ";
         for( int d = 0; d < meshDimensions; d++ )
            outputFile << " " << point[ d ];
         outputFile << endl;
      }

      typedef typename MeshType::template EntitiesTraits< meshDimensions >::GlobalIndexType CellIndexType;
      typedef typename MeshType::template EntitiesTraits< meshDimensions >::Type            CellType;
      typedef typename CellType::LocalIndexType                                             LocalIndexType;

      const CellIndexType numberOfCells = mesh.template getNumberOfEntities< meshDimensions >();
      outputFile << numberOfCells << endl;
      for( CellIndexType cellIdx = 0; cellIdx < numberOfCells; cellIdx++ )
      {
         const CellType& cell = mesh.template getEntity< meshDimensions >( cellIdx );
         outputFile << "   1";
         for( LocalIndexType cellVertexIdx = 0;
              cellVertexIdx < meshDimensions + 1;
              cellVertexIdx++ )
            outputFile << " " << cell.getVertexIndex( cellVertexIdx );
         outputFile << endl;
      }

      return true;
   }
};


#endif /* TNLMESHWRITERNETGEN_H_ */
