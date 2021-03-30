#include <TNL/Meshes/TypeResolver/resolveMeshType.h>

// Define the tag for the MeshTypeResolver configuration
struct MeshConfigTag {};

namespace TNL {
namespace Meshes {
namespace BuildConfigTags {

template<> struct MeshCellTopologyTag< MeshConfigTag, Topologies::Triangle > { enum { enabled = true }; };
template<> struct MeshCellTopologyTag< MeshConfigTag, Topologies::Quadrangle > { enum { enabled = true }; };

} // namespace BuildConfigTags
} // namespace Meshes
} // namespace TNL

// Define the main task/function of the program
template< typename Mesh >
bool task( const Mesh& mesh, const std::string& inputFileName )
{
   std::cout << "The file '" << inputFileName << "' contains the following mesh: "
             << TNL::getType<Mesh>() << std::endl;
   return true;
}

int main( int argc, char* argv[] )
{
   const std::string inputFileName = "example-triangles.vtu";

   auto wrapper = [&] ( auto& reader, auto&& mesh ) -> bool
   {
      return task( mesh, inputFileName );
   };
   return ! TNL::Meshes::resolveAndLoadMesh< MeshConfigTag, TNL::Devices::Host >( wrapper, inputFileName, "auto" );
}
