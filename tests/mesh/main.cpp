#include <iostream>
#include <mesh/config.h>
#include <mesh/mesh.h>
#include "mesh_info.h"
#include "test_configs.h"
#include <tnlConfig.h>

using namespace std;

const char lineVtkFile[] = TNL_SOURCE_DIRECTORY "tests/mesh/share/line.vtk";
const char twoTrianglesVtkFile[] = TNL_SOURCE_DIRECTORY "tests/mesh/share/2triangles.vtk";
const char twoSquaresVtkFile[] = TNL_SOURCE_DIRECTORY "tests/mesh/share/2squares.vtk";
const char twoCubesVtkFile[] = TNL_SOURCE_DIRECTORY "tests/mesh/share/2cubes.vtk";

const char lineVtkOutFile[] = TNL_TESTS_DIRECTORY "line-out.vtk";
const char twoTrianglesVtkOutFile[] = TNL_TESTS_DIRECTORY "2triangles-out.vtk";
const char twoSquaresVtkOutFile[] = TNL_TESTS_DIRECTORY "2squares-out.vtk";
const char twoCubesVtkOutFile[] = TNL_TESTS_DIRECTORY "2cubes-out.vtk";

const char lineMeshOutFile[] = TNL_TESTS_DIRECTORY "mesh/line-out.mesh";
const char twoTrianglesMeshOutFile[] = TNL_TESTS_DIRECTORY "mesh/2triangles-out.mesh";
const char twoSquaresMeshOutFile[] = TNL_TESTS_DIRECTORY "mesh/2squares-out.mesh";
const char twoCubesMeshOutFile[] = TNL_TESTS_DIRECTORY "mesh/2cubes-out.mesh";


template<typename MeshType>
void test(MeshType &mesh)
{
	cout << "Mesh memory consumption: " << mesh.memoryRequirement() << " B" << endl;

	MeshInfo<MeshType>::print(cout, mesh);

	cout << "===========================================================" << endl << endl;
}

template<typename MeshType>
void testWithoutOutput(MeshType &mesh)
{
	cout << "Mesh memory consumption: " << mesh.memoryRequirement() << " B" << endl;

	ostream stream(0);
	MeshInfo<MeshType>::print(stream, mesh);

	cout << "===========================================================" << endl << endl;
}

int main()
{
   cout << "Testing linear mesh ... " << endl;
	Mesh<LinearMeshConfig> mesh0;
	mesh0.load( lineVtkFile);
	test(mesh0);

	cout << "Testing 3D linear mesh with coborders 0->1..." << endl;
	Mesh<config::Lin3DConfig> testmesh0;
	testmesh0.load( lineVtkFile );
	testWithoutOutput(testmesh0);

	cout << "Testing TriangularMeshNoEdges..." << endl;
	Mesh<TriangularMeshNoEdgesConfig> mesh1;
	mesh1.load( twoTrianglesVtkFile );
	test(mesh1);

	cout << "Testing TriangularMesh..." << endl;
	Mesh<TriangularMeshConfig> mesh2;
	mesh2.load( twoTrianglesVtkFile );
	test(mesh2);

	cout << "Testing 2D triangular mesh with coborders 0->1, 0->2, 1->2..." << endl;
	Mesh<config::Tri2DAllConfig> testmesh1;
	testmesh1.load( twoTrianglesVtkFile );
	testWithoutOutput(testmesh1);

	cout << "Testing 2D quadrilateral mesh without borders 2->1, with coborders 1->2..." << endl;
	Mesh<config::Quadri2DCoborderWithoutBorderConfig> testmesh2;
	testmesh2.load( twoTrianglesVtkFile );
	testWithoutOutput(testmesh2);

	cout << "Testing HexahedralMesh..." << endl;
	Mesh<HexahedralMeshConfig> mesh3;
	mesh3.load( twoCubesVtkFile );
	testWithoutOutput(mesh3);

	cout << "Testing hexahedral mesh with nothing..." << endl;
	Mesh<config::Hexa3DNothingConfig> testmesh3;
	testmesh3.load( twoCubesVtkFile );
	testWithoutOutput(testmesh3);

	cout << "Testing hexahedral mesh with all coborders..." << endl;
	Mesh<config::Hexa3DAllConfig> testmesh4;
	testmesh4.load( twoCubesVtkFile );
	test(testmesh4);

	cout << "Testing IO... " << flush;
	mesh0.write( lineVtkOutFile );
	mesh0.write( lineMeshOutFile );
	mesh0.load( lineMeshOutFile );
	mesh1.write( twoTrianglesVtkOutFile );
	mesh1.write( twoTrianglesMeshOutFile );
	mesh1.load( twoTrianglesMeshOutFile );
	mesh3.write( twoCubesVtkOutFile );
	mesh3.write( twoCubesMeshOutFile );
	mesh3.load( twoCubesMeshOutFile );
	testmesh2.write( twoSquaresVtkOutFile );
	testmesh2.write( twoSquaresMeshOutFile );
	testmesh2.load( twoSquaresMeshOutFile );

	Mesh<TriangularMeshConfig> trimesh;
	trimesh.load("../tests/data/squarehole.mesh");
	trimesh.write("../tests/data/squarehole_out.vtk");
	trimesh.load("../tests/data/canyon2d.mesh");
	trimesh.write("../tests/data/canyon2d_out.vtk");
	trimesh.load("../tests/data/2triangles.tri");
	trimesh.write("../tests/data/2triangles_out.tri");

	Mesh<TetrahedralMeshConfig> tetramesh;
	tetramesh.load("../tests/data/cone.mesh");
	tetramesh.write("../tests/data/cone_out.vtk");
	tetramesh.load("../tests/data/canyon3d.mesh");
	tetramesh.write("../tests/data/canyon3d_out.vtk");

	cout << "done" << endl;

	return 0;
}
