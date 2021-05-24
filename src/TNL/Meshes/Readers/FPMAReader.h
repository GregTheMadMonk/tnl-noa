#pragma once

#include <fstream>
#include <sstream>
#include <vector>

#include <TNL/Meshes/Readers/MeshReader.h>

namespace TNL {
namespace Meshes {
namespace Readers {

class FPMAReader
: public MeshReader
{
public:
   FPMAReader() = default;

   FPMAReader( const std::string& fileName )
   : MeshReader( fileName )
   {}

   virtual void detectMesh() override
   {
      reset();

      std::ifstream inputFile( fileName );
      if( ! inputFile )
         throw MeshReaderError( "FPMAReader", "failed to open the file '" + fileName + "'." );

      std::string line;
      std::istringstream iss;

      // fpma format doesn't provide types
      pointsType = "double";
      connectivityType = offsetsType = "std::int32_t";

      // it is expected, that fpma format always stores polyhedral mesh
      spaceDimension = meshDimension = 3;
      cellShape = VTK::EntityShape::Polyhedron;

      // arrays holding the data from the file
      std::vector< double > pointsArray;
      std::vector< std::int32_t > cellConnectivityArray, cellOffsetsArray;
      std::vector< std::int32_t > faceConnectivityArray, faceOffsetsArray;

      // read number of points
      nextLine( inputFile, iss, line );
      iss >> NumberOfPoints;

      // read points
      nextLine( inputFile, iss, line );
      for( std::size_t pointIndex = 0; pointIndex < NumberOfPoints; pointIndex++ ) {
         if( ! inputFile ) {
            reset();
            throw MeshReaderError( "FPMAReader", "unable to read enough vertices, the file may be invalid or corrupted." );
         }

         // read the coordinates of a point
         for( int i = 0; i < 3; i++ ) {
            double aux;
            iss >> aux;
            if( ! iss ) {
               reset();
               throw MeshReaderError( "FPMAReader", "unable to read " + std::to_string(i) + "th component of the vertex number " + std::to_string(pointIndex) + "." );
            }

            pointsArray.push_back( aux );
         }
      }

      // read number of faces
      nextLine( inputFile, iss, line );
      iss >> NumberOfFaces;

      // read faces
      for( std::size_t faceIndex = 0; faceIndex < NumberOfFaces; faceIndex++ ) {
         if( ! inputFile ) {
            reset();
            throw MeshReaderError( "FPMAReader", "unable to read enough faces, the file may be invalid or corrupted." );
         }

         // read number of points of a face
         size_t numberOfFacePoints;
         nextLine( inputFile, iss, line );
         iss >> numberOfFacePoints;

         // read points of a face
         for( std::size_t i = 0; i < numberOfFacePoints; i++ ) {
            if( ! iss ) {
               reset();
               throw MeshReaderError( "FPMAReader", "unable to read " + std::to_string(i) + "th component of the face number " + std::to_string(faceIndex) + "." );
            }

            std::uint32_t pointIndex;
            iss >> pointIndex;

            if( ! iss || pointIndex >= NumberOfPoints ) {
               reset();
               throw MeshReaderError( "FPMAReader", std::to_string(i) + "th component of the face number " + std::to_string(faceIndex) + " is invalid." );
            }

            faceConnectivityArray.emplace_back( pointIndex );
         }

         faceOffsetsArray.emplace_back( faceConnectivityArray.size() );
      }

      // read number of cells
      nextLine( inputFile, iss, line );
      iss >> NumberOfCells;

      // read faces
      for( std::size_t cellIndex = 0; cellIndex < NumberOfCells; cellIndex++ ) {
         if( ! inputFile ) {
            reset();
            throw MeshReaderError( "FPMAReader", "unable to read enough cells, the file may be invalid or corrupted." );
         }

         // read number of faces of a cell
         size_t numberOfCellFaces;
         nextLine( inputFile, iss, line );
         iss >> numberOfCellFaces;

         // read faces of a cell
         for( std::size_t i = 0; i < numberOfCellFaces; i++ ) {
            if( ! iss ) {
               reset();
               throw MeshReaderError( "FPMAReader", "unable to read " + std::to_string(i) + "th component of the cell number " + std::to_string(cellIndex) + "." );
            }

            std::uint32_t faceIndex;
            iss >> faceIndex;

            if( ! iss || faceIndex >= NumberOfFaces ) {
               reset();
               throw MeshReaderError( "FPMAReader", std::to_string(i) + "th component of the cell number " + std::to_string(faceIndex) + " is invalid." );
            }

            cellConnectivityArray.emplace_back( faceIndex );
         }

         cellOffsetsArray.emplace_back( cellConnectivityArray.size() );
      }

      // set the arrays to the base class
      this->pointsArray = std::move(pointsArray);
      this->cellConnectivityArray = std::move(cellConnectivityArray);
      this->cellOffsetsArray = std::move(cellOffsetsArray);
      this->faceConnectivityArray = std::move(faceConnectivityArray);
      this->faceOffsetsArray = std::move(faceOffsetsArray);

      // indicate success by setting the mesh type
      meshType = "Meshes::Mesh";
   }
private:
   void nextLine( std::ifstream& inputFile, std::istringstream& iss, std::string& line )
   {
      iss.clear();
      // get next non-empty line, that isn't a comment
      while( std::getline( inputFile, line ) && ( line.empty() || line[0] == '#' ) ) {}
      iss.str( line );
   }
};

} // namespace Readers
} // namespace Meshes
} // namespace TNL
