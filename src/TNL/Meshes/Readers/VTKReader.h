/***************************************************************************
                          VTKReader.h  -  description
                             -------------------
    begin                : Feb 25, 2017
    copyright            : (C) 2017 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <fstream>
#include <sstream>
#include <vector>

#include <TNL/Meshes/MeshBuilder.h>
#include <TNL/Meshes/VTKTraits.h>
#include <TNL/Exceptions/NotImplementedError.h>

namespace TNL {
namespace Meshes {
namespace Readers {

class VTKReader
{
public:
   VTKReader() = delete;

   VTKReader( const String& fileName )
   : fileName( fileName )
   {}

   bool detectMesh()
   {
      this->reset();

      std::ifstream inputFile( fileName.getString() );
      if( ! inputFile ) {
         std::cerr << "Failed to open the file " << fileName << "." << std::endl;
         return false;
      }

      if( ! parseHeader( inputFile ) )
         return false;
      const auto positionAfterHeading = inputFile.tellg();

      if( dataset != "UNSTRUCTURED_GRID" ) {
         std::cerr << "VTKReader: the dataset '" << dataset << "' is not supported." << std::endl;
         return false;
      }

      // TODO: implement binary parsing
      if( dataType == "BINARY" ) {
         throw Exceptions::NotImplementedError("VTKReader: parsing of BINARY data is not implemented yet.");
      }

      std::string line, aux;
      std::istringstream iss;

      // parse points section
      if( ! findSection( inputFile, "POINTS" ) ) {
         std::cerr << "VTKReader: unable to find the POINTS section, the file may be invalid or corrupted." << std::endl;
         return false;
      }
      getline( inputFile, line );
      iss.clear();
      iss.str( line );
      iss >> aux;
      iss >> NumberOfPoints;
      iss >> realType;

      // read points
      worldDimension = 0;
      for( std::size_t pointIndex = 0; pointIndex < NumberOfPoints; pointIndex++ ) {
         if( ! inputFile ) {
            std::cerr << "VTKReader: unable to read enough vertices, the file may be invalid or corrupted." << std::endl;
            return false;
         }
         getline( inputFile, line );

         // read the coordinates and compute the world dimension
         iss.clear();
         iss.str( line );
         for( int i = 0; i < 3; i++ ) {
            double aux;
            iss >> aux;
            if( ! iss ) {
               std::cerr << "VTKReader: unable to read " << i << "th component of the vertex number " << pointIndex << "." << std::endl;
               return false;
            }
            if( aux != 0.0 )
               worldDimension = std::max( worldDimension, i + 1 );
            pointsArray.push_back( aux );
         }
      }

      // skip to the CELL_TYPES section
      if( ! findSection( inputFile, "CELL_TYPES" ) ) {
         std::cerr << "VTKReader: unable to find the CELL_TYPES section, the file may be invalid or corrupted." << std::endl;
         return false;
      }
      getline( inputFile, line );
      iss.clear();
      iss.str( line );
      iss >> aux;
      std::size_t NumberOfEntities;
      iss >> NumberOfEntities;

      // read entity types
      for( std::size_t entityIndex = 0; entityIndex < NumberOfEntities; entityIndex++ ) {
         if( ! inputFile ) {
            std::cerr << "VTKReader: unable to read enough cell types, the file may be invalid or corrupted." << std::endl;
            this->reset();
            return false;
         }
         getline( inputFile, line );

         // get entity type
         int typeId;
         iss.clear();
         iss.str( line );
         iss >> typeId;
         typesArray.push_back( typeId );
      }

      // count entities for each dimension
      std::size_t entitiesCounts[4] = {0, 0, 0, 0};
      for( auto c : typesArray ) {
         const int dimension = getEntityDimension( (VTK::EntityShape) c );
         ++entitiesCounts[dimension];
      }

      // set meshDimension
      meshDimension = 3;
      if( entitiesCounts[3] == 0 ) {
         meshDimension--;
         if( entitiesCounts[2] == 0 ) {
            meshDimension--;
            if( entitiesCounts[1] == 0 )
               meshDimension--;
         }
      }

      if( meshDimension == 0 ) {
         std::cerr << "Mesh dimension cannot be 0. Are there any entities at all?" << std::endl;
         this->reset();
         return false;
      }

      // filter out cell shapes
      std::vector< std::uint8_t > cellTypes;
      for( auto c : typesArray ) {
         const int dimension = getEntityDimension( (VTK::EntityShape) c );
         if( dimension == meshDimension )
            cellTypes.push_back( c );
      }

      // set number of cells
      NumberOfCells = cellTypes.size();
      if( NumberOfCells == 0 || NumberOfCells != entitiesCounts[meshDimension] ) {
         std::cerr << "VTKReader: invalid number of cells (" << NumberOfCells << "). Counts of entities for each dimension (0,1,2,3) are: ";
         std::cerr << entitiesCounts[0] << ", " << entitiesCounts[1] << ", " << entitiesCounts[2] << ", " << entitiesCounts[3] << std::endl;
         this->reset();
         return false;
      }

      // validate cell types
      cellShape = (VTK::EntityShape) cellTypes[0];
      for( auto c : cellTypes )
         if( (VTK::EntityShape) c != cellShape ) {
            std::cerr << "Mixed unstructured meshes are not supported. There are cells with type "
                      << VTK::getShapeName(cellShape) + " and " + VTK::getShapeName((VTK::EntityShape) c) << "." << std::endl;
            this->reset();
            return false;
         }

      // find to the CELLS section
      if( ! findSection( inputFile, "CELLS", positionAfterHeading ) ) {
         std::cerr << "VTKReader: unable to find the CELLS section, the file may be invalid or corrupted." << std::endl;
         this->reset();
         return false;
      }
      getline( inputFile, line );

      // read entities
      for( std::size_t entityIndex = 0; entityIndex < NumberOfEntities; entityIndex++ ) {
         if( ! inputFile ) {
            std::cerr << "VTKReader: unable to read enough cells, the file may be invalid or corrupted.";
            std::cerr << " (entityIndex = " << entityIndex << ")" << std::endl;
            this->reset();
            return false;
         }
         getline( inputFile, line );

         if( (VTK::EntityShape) typesArray[ entityIndex ] == cellShape ) {
            iss.clear();
            iss.str( line );
            // read number of subvertices
            int subvertices = 0;
            iss >> subvertices;
            for( int v = 0; v < subvertices; v++ ) {
               std::size_t vid;
               iss >> vid;
               if( ! iss ) {
                  std::cerr << "VTKReader: unable to read enough cells, the file may be invalid or corrupted.";
                  std::cerr << " (entityIndex = " << entityIndex << ", subvertex = " << v << ")" << std::endl;
                  std::cerr << line << std::endl;
                  this->reset();
                  return false;
               }
               connectivityArray.push_back( vid );
            }
            offsetsArray.push_back( connectivityArray.size() );
         }
      }

      // set cell types
      std::swap( cellTypes, typesArray );

      meshDetected = true;
      return true;
   }

   template< typename MeshType >
   bool readMesh( MeshType& mesh )
   {
      // check that detectMesh has been called
      if( ! meshDetected )
         detectMesh();

      using MeshBuilder = MeshBuilder< MeshType >;
      using PointType = typename MeshType::PointType;
      using CellSeedType = typename MeshBuilder::CellSeedType;

      MeshBuilder meshBuilder;
      meshBuilder.setPointsCount( NumberOfPoints );
      meshBuilder.setCellsCount( NumberOfCells );

      // assign points
      PointType p;
      std::size_t i = 0;
      for( auto c : pointsArray ) {
         int dim = i++ % 3;
         if( dim >= PointType::getSize() )
            continue;
         p[dim] = c;
         if( dim == PointType::getSize() - 1 )
            meshBuilder.setPoint( (i - 1) / 3, p );
      }

      // assign cells
      std::size_t offsetStart = 0;
      for( std::size_t i = 0; i < NumberOfCells; i++ ) {
         CellSeedType& seed = meshBuilder.getCellSeed( i );
         const std::size_t offsetEnd = offsetsArray[ i ];
         for( std::size_t o = offsetStart; o < offsetEnd; o++ )
            seed.setCornerId( o - offsetStart, connectivityArray[ o ] );
         offsetStart = offsetEnd;
      }

      // reset arrays since they are not needed anymore
      pointsArray = {};
      connectivityArray = offsetsArray = {};
      typesArray = {};

      return meshBuilder.build( mesh );
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
      return realType.c_str();
   }

   String
   getGlobalIndexType() const
   {
      // not stored in the VTK file
      return "int";
   }

   String
   getLocalIndexType() const
   {
      // not stored in the VTK file
      return "short int";
   }

protected:
   // output of parseHeader
   std::string dataType;
   std::string dataset;

   String fileName;
   bool meshDetected = false;

   std::size_t NumberOfPoints, NumberOfCells;
   int meshDimension, worldDimension;
   VTK::EntityShape cellShape = VTK::EntityShape::Vertex;
   std::string realType;

   // arrays holding the data from the VTK file
   std::vector< double > pointsArray;
   std::vector< std::int64_t > connectivityArray, offsetsArray;
   std::vector< std::uint8_t > typesArray;

   void reset()
   {
      meshDetected = false;
      NumberOfPoints = NumberOfCells = 0;
      meshDimension = worldDimension = 0;
      cellShape = VTK::EntityShape::Vertex;
      realType = "";
      // reset arrays
      pointsArray = {};
      connectivityArray = offsetsArray = {};
      typesArray = {};
   }

   bool parseHeader( std::istream& str )
   {
      std::string line;
      std::istringstream iss;

      // check header
      getline( str, line );
      if( line != "# vtk DataFile Version 2.0" ) {
         std::cerr << "VTKReader: unsupported VTK header: '" << line << "'." << std::endl;
         return false;
      }

      // skip title
      if( ! str )
         return false;
      getline( str, line );

      // parse data type
      if( ! str )
         return false;
      getline( str, dataType );
      if( dataType != "ASCII" && dataType != "BINARY" ) {
         std::cerr << "VTKReader: unknown data type: '" << dataType << "'." << std::endl;
         return false;
      }

      // parse dataset
      if( ! str )
         return false;
      getline( str, line );
      iss.clear();
      iss.str( line );
      std::string tmp;
      iss >> tmp;
      if( tmp != "DATASET" ) {
         std::cerr << "VTKReader: wrong dataset specification: '" << line << "'." << std::endl;
         return false;
      }
      iss >> dataset;

      return true;
   }

   bool findSection( std::istream& str, const std::string& section, std::ios::pos_type begin = -1 )
   {
      std::string line, aux;
      std::istringstream iss;

      if( begin >= 0 )
         str.seekg( begin );

      while( str ) {
         std::ios::pos_type currentPosition = str.tellg();
         getline( str, line );
         iss.clear();
         iss.str( line );
         iss >> aux;
         if( aux == section ) {
            str.seekg( currentPosition );
            return true;
         }
      }

      return false;
   }
};

} // namespace Readers
} // namespace Meshes
} // namespace TNL
