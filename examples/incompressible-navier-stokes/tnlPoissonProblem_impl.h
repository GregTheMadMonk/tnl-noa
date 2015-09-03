
#ifndef TNLPOISSONPROBLEM_IMP_H
#define	TNLPOISSONPROBLEM_IMP_H

#include "tnlPoissonProblem.h"
#include <mesh/tnlGrid.h>



template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
template< typename Vector >
#ifdef HAVE_CUDA
__device__ __host__
#endif
Real
tnlPoissonProblem< tnlGrid< 2, MeshReal, Device, MeshIndex >, Real, Index >::
getValue( const MeshType& mesh,
          const IndexType cellIndex,
          const CoordinatesType& coordinates,
          const Vector& u,
          const Real& time ) const
{
   return ( u[ mesh.getCellXPredecessor( cellIndex ) ]
            - 2.0 * u[ cellIndex ]
            + u[ mesh.getCellXSuccessor( cellIndex ) ] ) * mesh.getHxSquareInverse() +
           ( u[ mesh.getCellYPredecessor( cellIndex ) ]
             - 2.0 * u[ cellIndex ]
             + u[ mesh.getCellYSuccessor( cellIndex ) ] ) * mesh.getHySquareInverse();
}



#endif	/* TNLPOISSONPROBLEM_IMP_H */
