\page tutorial_Meshes  Unstructured meshes tutorial

[TOC]

## Introduction

The [Mesh](@ref TNL::Meshes::Mesh) class template is a data structure for _conforming unstructured homogeneous_ meshes, which can be used as the fundamental data structure for numerical schemes based on finite volume or finite element methods.
The abstract representation supports almost any cell shape which can be described by an [entity topology](@ref TNL::Meshes::Topologies).
Currently there are common 2D quadrilateral, 3D hexahedron and arbitrarily dimensional simplex topologies built in the library.
The implementation is highly configurable via templates of the C++ language, which allows to avoid the storage of unnecessary dynamic data.
The internal memory layout is based on state--of--the--art [sparse matrix formats](@ref TNL::Matrices), which are optimized for different hardware architectures in order to provide high performance computations.
The [DistributedMesh](@ref TNL::Meshes::DistributedMeshes::DistributedMesh) class template is an extended data structure based on `Mesh`, which allows to represent meshes decomposed into several subdomains for distributed computing using the Message Passing Interface (MPI).

## Reading a mesh from a file

The most common way of mesh initialization is by reading a prepared input file created by an external program.
TNL provides classes and functions for reading the common VTK, VTU and Netgen file formats.

\dontinclude ReadMeshExample.cpp

The main difficulty is mapping the mesh included in the file to the correct C++ type, which can represent the mesh stored in the file.
This can be done with the [MeshTypeResolver](@ref TNL::Meshes::MeshTypeResolver) class, which needs to be configured to enable the processing of the specific cell topologies, which we want our program to handle.
For example, in the following code we enable loading of 2D triangular and quadrangular meshes:

\skip #include
\until // namespace TNL

There are other build config tags which can be used to enable or disable specific types used in the mesh: `RealType`, `GlobalIndexType` and `LocalIndexType`.
See the [BuildConfigTags](@ref TNL::Meshes::BuildConfigTags) namespace for an overview of these tags.

Next, we can define the main task of our program as a templated function, which will be ultimately launched with the correct mesh type based on the input file.
We can also use any number of additional parameters, such as the input file name:

\skip Define the main task
\until }

Of course in practice, the function would be much more complex than this example, where we just print the file name and some textual representation of the mesh to the standard output.

Finally, we define the `main` function, which sets the input parameters (hard-coded in this example) and calls the [resolveAndLoadMesh](@ref TNL::Meshes::resolveAndLoadMesh) function to resolve the mesh type and load the mesh from the file into the created object:

\skip int main
\until }
\until }

We need to specify two template parameters when calling `resolveAndLoadMesh`:

1. our build config tag (`MeshConfigTag` in this example),
2. and the [device](@ref TNL::Devices) where the mesh should be allocated.

Then we pass the the function which should be called with the initialized mesh, the input file name, and the input file format (`"auto"` means auto-detection based on the file name).
In order to show the flexibility of passing other parameters to our main `task` function as defined above, we suggest to implement a wrapper lambda function (called `wrapper` in the example), which captures the relevant variables and forwards them to the `task`.

The return value of the `resolveAndLoadMesh` function is a boolean value representing the success (`true`) or failure (`false`) of the whole function call chain.
Hence, the return type of both `wrapper` and `task` needs to be `bool` as well.

For completeness, the full example follows:
\includelineno ReadMeshExample.cpp

## Mesh configuration

The [Mesh](@ref TNL::Meshes::Mesh) class template is configurable via its first template parameter, `Config`.
By default, the \ref TNL::Meshes::DefaultConfig template is used.
Alternative, user-specified configuration templates can be specified by defining the mesh configuration as the `MeshConfig` template in the [MeshConfigTemplateTag](@ref TNL::Meshes::BuildConfigTags::MeshConfigTemplateTag) build config tag specialization.
For example, here we derive the `MeshConfig` template from the [DefaultConfig](@ref TNL::Meshes::DefaultConfig) template and override the `subentityStorage` member function to store only those subentity incidence matrices, where the subentity dimension is 0 and the other dimension is at least $D-1$.
Hence, only faces and cells will be able to access their subvertices and there will be no other links from entities to their subentities.


```cpp
// Define the tag for the MeshTypeResolver configuration
struct MyConfigTag {};

namespace TNL {
namespace Meshes {
namespace BuildConfigTags {

// Create a template specialization of the tag specifying the MeshConfig template to use as the Config parameter for the mesh.
template<>
struct MeshConfigTemplateTag< MyConfigTag >
{
   template< typename Cell,
             int WorldDimension = Cell::dimension,
             typename Real = double,
             typename GlobalIndex = int,
             typename LocalIndex = GlobalIndex >
   struct MeshConfig
      : public DefaultConfig< Cell, WorldDimension, Real, GlobalIndex, LocalIndex >
   {
      template< typename EntityTopology >
      static constexpr bool subentityStorage( EntityTopology, int SubentityDimension )
      {
         return SubentityDimension == 0 && EntityTopology::dimension >= Cell::dimension - 1;
      }
   };
};

} // namespace BuildConfigTags
} // namespace Meshes
} // namespace TNL
```

## Public interface and basic usage

The whole public interface of the unstructured mesh and its mesh entity class can be found in the reference manual: \ref TNL::Meshes::Mesh, \ref TNL::Meshes::MeshEntity.

## Parallel iteration over mesh entities

## Writing a mesh and data to a file

## Example: Game of Life
