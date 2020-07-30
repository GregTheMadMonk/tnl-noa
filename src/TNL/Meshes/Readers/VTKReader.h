/***************************************************************************
                          VTKReader.h  -  description
                             -------------------
    begin                : Feb 25, 2017
    copyright            : (C) 2017 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

// Implemented by: Jakub Klinkovský

#pragma once

#include <fstream>
#include <sstream>
#include <vector>

#include <TNL/Meshes/Readers/MeshReader.h>
#include <TNL/Exceptions/NotImplementedError.h>

namespace TNL {
namespace Meshes {
namespace Readers {

class VTKReader
: public MeshReader
{
public:
   VTKReader() = default;

   VTKReader( const std::string& fileName )
   : MeshReader( fileName )
   {}

   virtual void detectMesh() override
   {
      reset();

      std::ifstream inputFile( fileName );
      if( ! inputFile )
         throw MeshReaderError( "VTKReader", "failed to open the file '" + fileName + "'." );

      if( ! parseHeader( inputFile ) )
         throw MeshReaderError( "VTKReader", "failed to parse the VTK file header." );
      const auto positionAfterHeading = inputFile.tellg();

      if( dataset != "UNSTRUCTURED_GRID" )
         throw MeshReaderError( "VTKReader", "the dataset '" + dataset + "' is not supported." );

      // TODO: implement binary parsing
      if( dataType == "BINARY" )
         throw Exceptions::NotImplementedError("VTKReader: parsing of BINARY data is not implemented yet.");

      std::string line, aux;
      std::istringstream iss;

      // parse points section
      if( ! findSection( inputFile, "POINTS" ) )
         throw MeshReaderError( "VTKReader", "unable to find the POINTS section, the file may be invalid or corrupted." );
      getline( inputFile, line );
      iss.clear();
      iss.str( line );
      iss >> aux;
      iss >> NumberOfPoints;
      iss >> pointsType;

      // global index type is not stored in legacy VTK files
      connectivityType = offsetsType = "std::int32_t";
      // only std::uint8_t makes sense for entity types
      typesType = "std::uint8_t";

      // arrays holding the data from the VTK file
      std::vector< double > pointsArray;
      std::vector< std::int32_t > connectivityArray, offsetsArray;
      std::vector< std::uint8_t > typesArray;

      // read points
      worldDimension = 0;
      for( std::size_t pointIndex = 0; pointIndex < NumberOfPoints; pointIndex++ ) {
         if( ! inputFile ) {
            reset();
            throw MeshReaderError( "VTKReader", "unable to read enough vertices, the file may be invalid or corrupted." );
         }
         getline( inputFile, line );

         // read the coordinates and compute the world dimension
         iss.clear();
         iss.str( line );
         for( int i = 0; i < 3; i++ ) {
            double aux;
            iss >> aux;
            if( ! iss ) {
               reset();
               throw MeshReaderError( "VTKReader", "unable to read " + std::to_string(i) + "th component of the vertex number " + std::to_string(pointIndex) + "." );
            }
            if( aux != 0.0 )
               worldDimension = std::max( worldDimension, i + 1 );
            pointsArray.push_back( aux );
         }
      }

      // skip to the CELL_TYPES section
      if( ! findSection( inputFile, "CELL_TYPES" ) ) {
         reset();
         throw MeshReaderError( "VTKReader", "unable to find the CELL_TYPES section, the file may be invalid or corrupted." );
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
            reset();
            throw MeshReaderError( "VTKReader", "unable to read enough cell types, the file may be invalid or corrupted." );
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
         reset();
         throw MeshReaderError( "VTKReader", "Mesh dimension cannot be 0. Are there any entities at all?" );
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
         const std::string msg = "invalid number of cells (" + std::to_string(NumberOfCells) + "). Counts of entities for each dimension (0,1,2,3) are: "
                               + std::to_string(entitiesCounts[0]) + ", " + std::to_string(entitiesCounts[1]) + ", "
                               + std::to_string(entitiesCounts[2]) + ", " + std::to_string(entitiesCounts[3]);
         reset();
         throw MeshReaderError( "VTKReader", msg );
      }

      // validate cell types
      cellShape = (VTK::EntityShape) cellTypes[0];
      for( auto c : cellTypes )
         if( (VTK::EntityShape) c != cellShape ) {
            const std::string msg = "Mixed unstructured meshes are not supported. There are cells with type "
                                  + VTK::getShapeName(cellShape) + " and " + VTK::getShapeName((VTK::EntityShape) c) + ".";
            reset();
            throw MeshReaderError( "VTKReader", msg );
         }

      // find to the CELLS section
      if( ! findSection( inputFile, "CELLS", positionAfterHeading ) ) {
         reset();
         throw MeshReaderError( "VTKReader", "unable to find the CELLS section, the file may be invalid or corrupted." );
      }
      getline( inputFile, line );

      // read entities
      for( std::size_t entityIndex = 0; entityIndex < NumberOfEntities; entityIndex++ ) {
         if( ! inputFile ) {
            reset();
            throw MeshReaderError( "VTKReader", "unable to read enough cells, the file may be invalid or corrupted."
                                                " (entityIndex = " + std::to_string(entityIndex) + ")" );
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
                  reset();
                  throw MeshReaderError( "VTKReader", "unable to read enough cells, the file may be invalid or corrupted."
                                                      " (entityIndex = " + std::to_string(entityIndex) + ", subvertex = " + std::to_string(v) + ")" );
               }
               connectivityArray.push_back( vid );
            }
            offsetsArray.push_back( connectivityArray.size() );
         }
      }

      // set cell types
      std::swap( cellTypes, typesArray );

      // set the arrays to the base class
      this->pointsArray = std::move(pointsArray);
      this->connectivityArray = std::move(connectivityArray);
      this->offsetsArray = std::move(offsetsArray);
      this->typesArray = std::move(typesArray);

      // indicate success by setting the mesh type
      meshType = "Meshes::Mesh";
   }

   virtual void reset() override
   {
      resetBase();
      dataType = "";
      dataset = "";
   }

protected:
   // output of parseHeader
   std::string dataType;
   std::string dataset;

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
