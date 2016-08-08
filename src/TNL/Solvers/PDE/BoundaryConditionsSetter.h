/***************************************************************************
                          BoundaryConditionsSetter.h  -  description
                             -------------------
    begin                : Dec 30, 2015
    copyright            : (C) 2015 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */


#pragma once

#include <TNL/Devices/Cuda.h>
#include <TNL/Functions/FunctionAdapter.h>

namespace TNL {
namespace Solvers {
namespace PDE {

template< typename Real,
          typename DofVector,
          typename BoundaryConditions >
class BoundaryConditionsSetterTraverserUserData
{
   public:

      const Real *time;

      const BoundaryConditions* boundaryConditions;

      DofVector *u;

      BoundaryConditionsSetterTraverserUserData(
         const Real& time,
         const BoundaryConditions& boundaryConditions,
         DofVector& u )
      : time( &time ),
        boundaryConditions( &boundaryConditions ),
        u( &u )
      {};
};


template< typename MeshFunction,
          typename BoundaryConditions >
class BoundaryConditionsSetter
{
   public:
      typedef typename MeshFunction::MeshType MeshType;
      typedef typename MeshFunction::RealType RealType;
      typedef typename MeshFunction::DeviceType DeviceType;
      typedef typename MeshFunction::IndexType IndexType;
      typedef BoundaryConditionsSetterTraverserUserData<
         RealType,
         MeshFunction,
         BoundaryConditions > TraverserUserData;

      template< typename EntityType = typename MeshType::Cell >
      static void apply( const BoundaryConditions& boundaryConditions,
                         const RealType& time,
                         MeshFunction& u );
 
      class TraverserBoundaryEntitiesProcessor
      {
         public:
 
            template< typename GridEntity >
            __cuda_callable__
            static inline void processEntity( const MeshType& mesh,
                                              TraverserUserData& userData,
                                              const GridEntity& entity )
            {
               ( *userData.u )( entity ) = userData.boundaryConditions->operator()
               ( *userData.u,
                 entity,
                 *userData.time );
            }

      };
};

} // namespace PDE
} // namespace Solvers
} // namespace TNL

#include <TNL/Solvers/PDE/BoundaryConditionsSetter_impl.h>


