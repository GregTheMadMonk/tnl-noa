#ifndef TNLTIMEFUNCTION_IMPL_H
#define	TNLTIMEFUNCTION_IMPL_H

#include "tnlTimeFunction.h"

template<>
template<int dim, typename Real, typename Device, typename Index>
typename tnlGrid<dim,Real,Device,Index,tnlIdenticalGridGeometry>::RealType 
TimeFunction<tnlGrid<dim,Real,Device,Index,tnlIdenticalGridGeometry>,TimeFunctionBase::TimeIndependent>::
getTimeValue(const RealType& time) const
{
      return 1.0;
}

template<>
template<int dim, typename Real, typename Device, typename Index>
typename tnlGrid<dim,Real,Device,Index,tnlIdenticalGridGeometry>::RealType 
TimeFunction<tnlGrid<dim,Real,Device,Index,tnlIdenticalGridGeometry>,TimeFunctionBase::Linear>::
getTimeValue(const RealType& time) const
{
      return time;
}

template<>
template<int dim, typename Real, typename Device, typename Index>
typename tnlGrid<dim,Real,Device,Index,tnlIdenticalGridGeometry>::RealType 
TimeFunction<tnlGrid<dim,Real,Device,Index,tnlIdenticalGridGeometry>,TimeFunctionBase::Quadratic>::
getTimeValue(const RealType& time) const
{
      return time*time;
}

template<>
template<int dim, typename Real, typename Device, typename Index>
typename tnlGrid<dim,Real,Device,Index,tnlIdenticalGridGeometry>::RealType 
TimeFunction<tnlGrid<dim,Real,Device,Index,tnlIdenticalGridGeometry>,TimeFunctionBase::Cosinus>::
getTimeValue(const RealType& time) const
{
      return cos(time);
}

template<>
template<int dim, typename Real, typename Device, typename Index>
typename tnlGrid<dim,Real,Device,Index,tnlIdenticalGridGeometry>::RealType 
TimeFunction<tnlGrid<dim,Real,Device,Index,tnlIdenticalGridGeometry>,TimeFunctionBase::TimeIndependent>::
getDerivation(const RealType& time) const
{
      return 0.0;
}

template<>
template<int dim, typename Real, typename Device, typename Index>
typename tnlGrid<dim,Real,Device,Index,tnlIdenticalGridGeometry>::RealType 
TimeFunction<tnlGrid<dim,Real,Device,Index,tnlIdenticalGridGeometry>,TimeFunctionBase::Linear>::
getDerivation(const RealType& time) const
{
      return 1.0;
}

template<>
template<int dim, typename Real, typename Device, typename Index>
typename tnlGrid<dim,Real,Device,Index,tnlIdenticalGridGeometry>::RealType 
TimeFunction<tnlGrid<dim,Real,Device,Index,tnlIdenticalGridGeometry>,TimeFunctionBase::Quadratic>::
getDerivation(const RealType& time) const
{
      return 2.0*time;
}

template<>
template<int dim, typename Real, typename Device, typename Index>
typename tnlGrid<dim,Real,Device,Index,tnlIdenticalGridGeometry>::RealType 
TimeFunction<tnlGrid<dim,Real,Device,Index,tnlIdenticalGridGeometry>,TimeFunctionBase::Cosinus>::
getDerivation(const RealType& time) const
{
      return -sin(time);
}

template<>
template<int dim, typename Real, typename Device, typename Index>
void TimeFunction<tnlGrid<dim,Real,Device,Index,tnlIdenticalGridGeometry>,TimeFunctionBase::TimeIndependent>::
applyInitTimeValues(SharedVectorType& u) const
{
   
}

template<>
template<int dim, typename Real, typename Device, typename Index>
void TimeFunction<tnlGrid<dim,Real,Device,Index,tnlIdenticalGridGeometry>,TimeFunctionBase::Linear>::
applyInitTimeValues(SharedVectorType& u) const
{
   #ifdef HAVE_OPENMP
      #pragma omp parallel for
   #endif
   for(IndexType i=1; i<(u.getSize()-1); i++)
   {         
     u[i] = 0;
   }
}

template<>
template<int dim, typename Real, typename Device, typename Index>
void TimeFunction<tnlGrid<dim,Real,Device,Index,tnlIdenticalGridGeometry>,TimeFunctionBase::Quadratic>::
applyInitTimeValues(SharedVectorType& u) const
{
   #ifdef HAVE_OPENMP
      #pragma omp parallel for
   #endif
   for(IndexType i=1; i<(u.getSize()-1); i++)
   {         
     u[i] = 0;
   }
}

template<>
template<int dim, typename Real, typename Device, typename Index>
void TimeFunction<tnlGrid<dim,Real,Device,Index,tnlIdenticalGridGeometry>,TimeFunctionBase::Cosinus>::
applyInitTimeValues(SharedVectorType& u) const
{

}

#endif	/* TNLTIMEFUNCTION_IMPL_H */
