/***************************************************************************
                          MeshWriterNetgen.h  -  description
                             -------------------
    begin                : Feb 22, 2014
    copyright            : (C) 2014 by Tomas Oberhuber
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
#include <iomanip>

namespace TNL {
namespace Meshes {

class MeshWriterNetgen
{
   public:

   template< typename MeshType >
   static bool writeMesh( const String& fileName,
                          MeshType& mesh,
                          bool verbose )
   {
      std::fstream outputFile;
      outputFile.open( fileName.getString(), std::ios::out );
      if( ! outputFile )
      {
         std::cerr << "I am not able to open the output file " << fileName << "." << std::endl;
         return false;
      }
      outputFile << std::setprecision( 6 );
      outputFile << fixed;

      const int meshDimension = MeshType::meshDimension;
      typedef typename MeshType::template EntitiesTraits< 0 >::GlobalIndexType VerticesIndexType;
      typedef typename MeshType::PointType                                     PointType;
      const VerticesIndexType numberOfVertices = mesh.getNumberOfVertices();
      outputFile << numberOfVertices << std::endl;
      for( VerticesIndexType i = 0; i < numberOfVertices; i++ )
      {
         const PointType& point = mesh.getVertex( i ).getPoint();
         outputFile << " ";
         for( int d = 0; d < meshDimension; d++ )
            outputFile << " " << point[ d ];
         outputFile << std::endl;
      }

      typedef typename MeshType::template EntitiesTraits< meshDimension >::GlobalIndexType CellIndexType;
      typedef typename MeshType::template EntitiesTraits< meshDimension >::Type            CellType;
      typedef typename CellType::LocalIndexType                                             LocalIndexType;

      const CellIndexType numberOfCells = mesh.template getNumberOfEntities< meshDimension >();
      outputFile << numberOfCells << std::endl;
      for( CellIndexType cellIdx = 0; cellIdx < numberOfCells; cellIdx++ )
      {
         const CellType& cell = mesh.template getEntity< meshDimension >( cellIdx );
         outputFile << "   1";
         for( LocalIndexType cellVertexIdx = 0;
              cellVertexIdx < meshDimension + 1;
              cellVertexIdx++ )
            outputFile << " " << cell.getVertexIndex( cellVertexIdx );
         outputFile << std::endl;
      }

      return true;
   }
};

} // namespace Meshes
} // namespace TNL
