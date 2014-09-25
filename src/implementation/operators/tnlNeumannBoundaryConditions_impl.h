#ifndef TNLNEUMANNBOUNDARYCONDITIONS_IMPL_H
#define	TNLNEUMANNBOUNDARYCONDITIONS_IMPL_H


template<typename Real, typename Device, typename Index>
template<typename AnalyticSpaceFunction, typename TimeFunction, typename Vector>
void tnlNeumannBoundaryConditions<tnlGrid<1,Real,Device,Index>>::
applyBoundaryConditions(const MeshType& mesh, Vector& u, const RealType& time, 
                        TimeFunction& timeFunction, AnalyticSpaceFunction& analyticSpaceFunction)
{
   
}

template<typename Real, typename Device, typename Index>
template<typename AnalyticSpaceFunction, typename TimeFunction, typename Vector>
void tnlNeumannBoundaryConditions<tnlGrid<1,Real,Device,Index>>::
applyBoundaryTimeDerivation(const MeshType& mesh, Vector& u, const RealType& time, 
                            TimeFunction& timeFunction, AnalyticSpaceFunction& analyticSpaceFunction)
{
   
}

template<typename Real, typename Device, typename Index>
template<typename AnalyticSpaceFunction, typename TimeFunction, typename Vector>
void tnlNeumannBoundaryConditions<tnlGrid<2,Real,Device,Index>>::
applyBoundaryConditions(const MeshType& mesh, Vector& u, const RealType& time, 
                        TimeFunction& timeFunction, AnalyticSpaceFunction& analyticSpaceFunction)
{
   
}

template<typename Real, typename Device, typename Index>
template<typename AnalyticSpaceFunction, typename TimeFunction, typename Vector>
void tnlNeumannBoundaryConditions<tnlGrid<2,Real,Device,Index>>::
applyBoundaryTimeDerivation(const MeshType& mesh, Vector& u, const RealType& time, 
                            TimeFunction& timeFunction, AnalyticSpaceFunction& analyticSpaceFunction)
{
   
}

template<typename Real, typename Device, typename Index>
template<typename AnalyticSpaceFunction, typename TimeFunction, typename Vector>
void tnlNeumannBoundaryConditions<tnlGrid<3,Real,Device,Index>>::
applyBoundaryConditions(const MeshType& mesh, Vector& u, const RealType& time, 
                        TimeFunction& timeFunction, AnalyticSpaceFunction& analyticSpaceFunction)
{
   
}

template<typename Real, typename Device, typename Index>
template<typename AnalyticSpaceFunction, typename TimeFunction, typename Vector>
void tnlNeumannBoundaryConditions<tnlGrid<3,Real,Device,Index>>::
applyBoundaryTimeDerivation(const MeshType& mesh, Vector& u, const RealType& time, 
                            TimeFunction& timeFunction, AnalyticSpaceFunction& analyticSpaceFunction)
{
   
}

#endif	/* TNLNEUMANNBOUNDARYCONDITIONS_IMPL_H */

