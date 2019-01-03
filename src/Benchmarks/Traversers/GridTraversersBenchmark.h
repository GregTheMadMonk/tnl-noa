/***************************************************************************
                          GridTraversersBenchmark.h  -  description
                             -------------------
    begin                : Dec 19, 2018
    copyright            : (C) 2018 by oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

// Implemented by: Tomas Oberhuber

#pragma once

#include <TNL/ParallelFor.h>
#include <TNL/Devices/Host.h>
#include <TNL/Devices/Cuda.h>
#include <TNL/Containers/Vector.h>
#include <TNL/Meshes/Grid.h>
#include <TNL/Meshes/GridEntityConfig.h>
#include <TNL/Meshes/Traverser.h>
#include <TNL/Functions/MeshFunction.h>
#include <TNL/Pointers/SharedPointer.h>
#include "cuda-kernels.h"

namespace TNL {
   namespace Benchmarks {
      namespace Traversers {

template< typename TraverserUserData >
class WriteOneEntitiesProcessor
{
   public:
      
      using MeshType = typename TraverserUserData::MeshType;
      using DeviceType = typename MeshType::DeviceType;

      template< typename GridEntity >
      __cuda_callable__
      static inline void processEntity( const MeshType& mesh,
                                        TraverserUserData& userData,
                                        const GridEntity& entity )
      {
         auto& u = userData.u.template modifyData< DeviceType >();
         u( entity ) += 1.0;
      }
};

template< typename MeshFunctionPointer >
class WriteOneUserData
{
   public:
      
      using MeshType = typename MeshFunctionPointer::ObjectType::MeshType;
      
      MeshFunctionPointer u;
};

template< int Dimension,
          typename Device,
          typename Real,
          typename Index >
class GridTraversersBenchmark{};

      } // namespace Traversers
   } // namespace Benchmarks
} // namespace TNL

#include "GridTraversersBenchmark_1D.h"
#include "GridTraversersBenchmark_2D.h"
#include "GridTraversersBenchmark_3D.h"