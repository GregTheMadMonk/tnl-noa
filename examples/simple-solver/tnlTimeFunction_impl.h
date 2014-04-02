#ifndef TNLTIMEFUNCTION_IMPL_H
#define	TNLTIMEFUNCTION_IMPL_H

#include "tnlTimeFunction.h"

template<>
template<int dim, typename Real, typename Device, typename Index>
typename tnlGrid<dim,Real,Device,Index,tnlIdenticalGridGeometry>::RealType 
TimeFunction<tnlGrid<dim,Real,Device,Index,tnlIdenticalGridGeometry>,TimeFunctionBase::TimeIndependent>::
getTimeValue(const RealType& time)
{
      return 1.0;
}

template<>
template<int dim, typename Real, typename Device, typename Index>
typename tnlGrid<dim,Real,Device,Index,tnlIdenticalGridGeometry>::RealType 
TimeFunction<tnlGrid<dim,Real,Device,Index,tnlIdenticalGridGeometry>,TimeFunctionBase::Linear>::
getTimeValue(const RealType& time)
{
      return time;
}

template<>
template<int dim, typename Real, typename Device, typename Index>
typename tnlGrid<dim,Real,Device,Index,tnlIdenticalGridGeometry>::RealType 
TimeFunction<tnlGrid<dim,Real,Device,Index,tnlIdenticalGridGeometry>,TimeFunctionBase::Quadratic>::
getTimeValue(const RealType& time)
{
      return time*time;
}

template<>
template<int dim, typename Real, typename Device, typename Index>
typename tnlGrid<dim,Real,Device,Index,tnlIdenticalGridGeometry>::RealType 
TimeFunction<tnlGrid<dim,Real,Device,Index,tnlIdenticalGridGeometry>,TimeFunctionBase::Cosinus>::
getTimeValue(const RealType& time)
{
      return cos(time);
}

template<>
template<int dim, typename Real, typename Device, typename Index>
typename tnlGrid<dim,Real,Device,Index,tnlIdenticalGridGeometry>::RealType 
TimeFunction<tnlGrid<dim,Real,Device,Index,tnlIdenticalGridGeometry>,TimeFunctionBase::TimeIndependent>::
getDerivation(const RealType& time)
{
      return 0.0;
}

template<>
template<int dim, typename Real, typename Device, typename Index>
typename tnlGrid<dim,Real,Device,Index,tnlIdenticalGridGeometry>::RealType 
TimeFunction<tnlGrid<dim,Real,Device,Index,tnlIdenticalGridGeometry>,TimeFunctionBase::Linear>::
getDerivation(const RealType& time)
{
      return 1.0;
}

template<>
template<int dim, typename Real, typename Device, typename Index>
typename tnlGrid<dim,Real,Device,Index,tnlIdenticalGridGeometry>::RealType 
TimeFunction<tnlGrid<dim,Real,Device,Index,tnlIdenticalGridGeometry>,TimeFunctionBase::Quadratic>::
getDerivation(const RealType& time)
{
      return 2.0*time;
}

template<>
template<int dim, typename Real, typename Device, typename Index>
typename tnlGrid<dim,Real,Device,Index,tnlIdenticalGridGeometry>::RealType 
TimeFunction<tnlGrid<dim,Real,Device,Index,tnlIdenticalGridGeometry>,TimeFunctionBase::Cosinus>::
getDerivation(const RealType& time)
{
      return -sin(time);
}

#endif	/* TNLTIMEFUNCTION_IMPL_H */

