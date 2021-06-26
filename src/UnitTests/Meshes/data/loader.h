#pragma once

#include <iostream>
#include <string>
#include <experimental/filesystem>

#ifndef TNL_MESH_TESTS_DATA_DIR
   #error "The TNL_MESH_TESTS_DATA_DIR macro is not defined."
#endif

template< typename MeshType, typename ReaderType >
MeshType loadMeshFromFile( std::string relative_path )
{
   namespace fs = std::experimental::filesystem;
   const fs::path full_path = fs::path( TNL_MESH_TESTS_DATA_DIR ) / fs::path( relative_path );
   std::cout << "Reading a mesh from file " << full_path << std::endl;

   MeshType mesh;
   ReaderType reader( full_path );
   reader.loadMesh( mesh );
   return mesh;
}
