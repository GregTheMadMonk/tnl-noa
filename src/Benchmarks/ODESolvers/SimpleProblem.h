/***************************************************************************
                          SimpleProblem.h  -  description
                             -------------------
    begin                : Jul 13, 2019
    copyright            : (C) 2019 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

// Implemented by: Tomas Oberhuber

#pragma once

#include <TNL/Devices/Host.h>
#include <TNL/Algorithms/ParallelFor.h>
#include <TNL/Communicators/MpiCommunicator.h>

namespace TNL {
   namespace Benchmarks {
      
template< typename Real = double,
   typename Device = Devices::Host,
   typename Index = int >
struct SimpleProblem
{
   using RealType = Real;
   using DeviceType = Device;
   using IndexType = Index;
   using DofVectorType = Containers::Vector< RealType, DeviceType, IndexType >;
   using CommunicatorType = Communicators::MpiCommunicator;
   
   template< typename VectorPointer >
   void getExplicitUpdate( const RealType& time,
      const RealType& tau,
      VectorPointer& _u,
      VectorPointer& _fu )
   {
      using VectorType = typename VectorPointer::ObjectType;
      using ViewType = typename VectorType::ViewType;
      auto u = _u->getView();
      auto fu = _fu->getView();
      auto computeF = [=] __cuda_callable__ ( IndexType i, ViewType& u, ViewType& fu )
      {
         fu[ i ] = 1.0;
      };
      Algorithms::ParallelFor< DeviceType >::exec( ( IndexType ) 0, u.getSize(), computeF, u, fu );
   }
   
   template< typename Vector >
   void applyBoundaryConditions( const RealType& t, Vector& u ) {};
      
};

   } // namespace Benchmarks
} // namespace TNL
