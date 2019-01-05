/***************************************************************************
                          BenchmarkTraverserUserData.h  -  description
                             -------------------
    begin                : Jan 5, 2019
    copyright            : (C) 2019 by oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

// Implemented by: Tomas Oberhuber

#pragma once

namespace TNL {
   namespace Benchmarks {
      namespace Traversers {

template< typename MeshFunction >
class BenchmarkTraverserUserData
{
   public:
      
      using MeshType = typename MeshFunction::MeshType;
      
      MeshFunction* u;
};


      } // namespace Traversers
   } // namespace Benchmarks
} // namespace TNL
