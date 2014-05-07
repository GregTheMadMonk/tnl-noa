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
   enum setTimeFunction{TimeIndependent};
   typedef tnlGrid<dim,Real,Device,Index,tnlIdenticalGridGeometry> MeshType;
   typedef typename MeshType::RealType RealType;
   typedef typename MeshType::DeviceType DeviceType;
   typedef typename MeshType::IndexType IndexType;
   typedef tnlSharedVector< RealType, DeviceType, IndexType > SharedVectorType;
   typedef tnlVector< RealType, DeviceType, IndexType> DofVectorType;
   
   RealType getTimeValue(const RealType& time) const;
   RealType getDerivation(const RealType& time) const;
   void applyInitTimeValues(SharedVectorType& u) const;
};

template<>
template<int dim, typename Real, typename Device, typename Index>
class TimeFunction<tnlGrid<dim,Real,Device,Index,tnlIdenticalGridGeometry>,TimeFunctionBase::Linear>: public TimeFunctionBase
{
   public:
   enum setTimeFunction{Linear};
   typedef tnlGrid<dim,Real,Device,Index,tnlIdenticalGridGeometry> MeshType;
   typedef typename MeshType::RealType RealType;
   typedef typename MeshType::DeviceType DeviceType;
   typedef typename MeshType::IndexType IndexType;
   typedef tnlSharedVector< RealType, DeviceType, IndexType > SharedVectorType;
   typedef tnlVector< RealType, DeviceType, IndexType> DofVectorType; 
   
   RealType getTimeValue(const RealType& time) const;
   RealType getDerivation(const RealType& time) const;
   void applyInitTimeValues(SharedVectorType& u) const;
};

template<>
template<int dim, typename Real, typename Device, typename Index>
class TimeFunction<tnlGrid<dim,Real,Device,Index,tnlIdenticalGridGeometry>,TimeFunctionBase::Quadratic>: public TimeFunctionBase
{
   public:
   enum setTimeFunction{Quadratic};
   typedef tnlGrid<dim,Real,Device,Index,tnlIdenticalGridGeometry> MeshType;
   typedef typename MeshType::RealType RealType;
   typedef typename MeshType::DeviceType DeviceType;
   typedef typename MeshType::IndexType IndexType;
   typedef tnlSharedVector< RealType, DeviceType, IndexType > SharedVectorType;
   typedef tnlVector< RealType, DeviceType, IndexType> DofVectorType;
   
   RealType getTimeValue(const RealType& time) const;
   RealType getDerivation(const RealType& time) const;
   void applyInitTimeValues(SharedVectorType& u) const;
};

template<>
template<int dim, typename Real, typename Device, typename Index>
class TimeFunction<tnlGrid<dim,Real,Device,Index,tnlIdenticalGridGeometry>,TimeFunctionBase::Cosinus>: public TimeFunctionBase
{
   public:
   enum setTimeFunction{Cosinus};
   typedef tnlGrid<dim,Real,Device,Index,tnlIdenticalGridGeometry> MeshType;
   typedef typename MeshType::RealType RealType;
   typedef typename MeshType::DeviceType DeviceType;
   typedef typename MeshType::IndexType IndexType;
   typedef tnlSharedVector< RealType, DeviceType, IndexType > SharedVectorType;
   typedef tnlVector< RealType, DeviceType, IndexType> DofVectorType;
   
   RealType getTimeValue(const RealType& time) const;
   RealType getDerivation(const RealType& time) const;
   void applyInitTimeValues(SharedVectorType& u) const;
};

#include "tnlTimeFunction_impl.h"

#endif	/* TNLTIMEFUNCTION_H */