#ifndef TNLTIMEFUNCTION_H
#define	TNLTIMEFUNCTION_H

class TimeFunctionBase
{
   public:
      
   enum TimeFunctionSet {TimeIndependent, Linear, Quadratic, Cosinus}; 
};

template<typename Mesh, TimeFunctionBase::TimeFunctionSet timefunction>
class TimeFunction
{
   
};

template<>
template<int dim, typename Real, typename Device, typename Index>
class TimeFunction<tnlGrid<dim,Real,Device,Index,tnlIdenticalGridGeometry>,TimeFunctionBase::TimeIndependent>: public TimeFunctionBase
{
   public:
   typedef tnlGrid<dim,Real,Device,Index,tnlIdenticalGridGeometry> MeshType;
   typedef typename MeshType::RealType RealType; 
   
   RealType getTimeValue(const RealType& time);
   RealType getDerivation(const RealType& time);
};

template<>
template<int dim, typename Real, typename Device, typename Index>
class TimeFunction<tnlGrid<dim,Real,Device,Index,tnlIdenticalGridGeometry>,TimeFunctionBase::Linear>: public TimeFunctionBase
{
   public:
   typedef tnlGrid<dim,Real,Device,Index,tnlIdenticalGridGeometry> MeshType;
   typedef typename MeshType::RealType RealType; 
   
   RealType getTimeValue(const RealType& time);
   RealType getDerivation(const RealType& time);
};

template<>
template<int dim, typename Real, typename Device, typename Index>
class TimeFunction<tnlGrid<dim,Real,Device,Index,tnlIdenticalGridGeometry>,TimeFunctionBase::Quadratic>: public TimeFunctionBase
{
   public:
   typedef tnlGrid<dim,Real,Device,Index,tnlIdenticalGridGeometry> MeshType;
   typedef typename MeshType::RealType RealType; 
   
   RealType getTimeValue(const RealType& time);
   RealType getDerivation(const RealType& time);
};

template<>
template<int dim, typename Real, typename Device, typename Index>
class TimeFunction<tnlGrid<dim,Real,Device,Index,tnlIdenticalGridGeometry>,TimeFunctionBase::Cosinus>: public TimeFunctionBase
{
   public:
   typedef tnlGrid<dim,Real,Device,Index,tnlIdenticalGridGeometry> MeshType;
   typedef typename MeshType::RealType RealType; 
   
   RealType getTimeValue(const RealType& time);
   RealType getDerivation(const RealType& time);
};

#include "tnlTimeFunction_impl.h"

#endif	/* TNLTIMEFUNCTION_H */
