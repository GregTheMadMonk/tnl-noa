
#ifndef TNLZERORIGHTHANDSIDE_IMPL_H
#define	TNLZERORIGHTHANDSIDE_IMPL_H

#include "tnlZeroRightHandSide.h"

template<typename Real, typename Device, typename Index>
template<typename AnalyticSpaceFunction, typename TimeFunction>
void tnlZeroRightHandSide<tnlGrid<1,Real,Device,Index,tnlIdenticalGridGeometry>>::
applyRHSValues(const MeshType& mesh, const RealType& time, DofVectorType& _fu,
               TimeFunction& timeFunction, AnalyticSpaceFunction& analyticSpaceFunction)
{           
   
}

template<typename Real, typename Device, typename Index>
template<typename AnalyticSpaceFunction, typename TimeFunction>
void tnlZeroRightHandSide<tnlGrid<2,Real,Device,Index,tnlIdenticalGridGeometry>>::
applyRHSValues(const MeshType& mesh, const RealType& time, DofVectorType& _fu,
               TimeFunction& timeFunction, AnalyticSpaceFunction& analyticSpaceFunction)
{           
   
}

template<typename Real, typename Device, typename Index>
template<typename AnalyticSpaceFunction, typename TimeFunction>
void tnlZeroRightHandSide<tnlGrid<3,Real,Device,Index,tnlIdenticalGridGeometry>>::
applyRHSValues(const MeshType& mesh, const RealType& time, DofVectorType& _fu,
               TimeFunction& timeFunction, AnalyticSpaceFunction& analyticSpaceFunction)
{           
   
}

#endif	/* TNLZERORIGHTHANDSIDE_IMPL_H */

