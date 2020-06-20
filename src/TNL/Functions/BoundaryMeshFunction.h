/***************************************************************************
                          BoundaryMeshFunction.h  -  description
                             -------------------
    begin                : Aug 21, 2018
    copyright            : (C) 2018 by oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/Functions/MeshFunction.h>

namespace TNL {
namespace Functions {   

   
// BoundaryMeshFunction is supposed to store values of a mesh functions only
// at boundary mesh entities. It is just a small memory optimization.
// Currently, it is only a wrap around common MeshFunction so that we can introduce
// boundary mesh functions in the rest of the code. 
// TODO: Implement it.
template< typename Mesh,
          int MeshEntityDimension = Mesh::getMeshDimension(),
          typename Real = typename Mesh::RealType >
class BoundaryMeshFunction :
   public MeshFunction< Mesh, MeshEntityDimension, Real >
{
   public:
      
      using BaseType = MeshFunction< Mesh, MeshEntityDimension, Real >;
      using typename BaseType::MeshType;
      using typename BaseType::DeviceType;
      using typename BaseType::IndexType;
      using typename BaseType::MeshPointer;
      using typename BaseType::RealType;
      using typename BaseType::VectorType;
};

} // namespace Functions
} // namespace TNL
