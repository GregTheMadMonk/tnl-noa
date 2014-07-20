#ifndef TNLTIMEFUNCTION_H
#define	TNLTIMEFUNCTION_H

class TimeFunctionBase
{
   public:
      
   enum TimeFunctionSet {TimeIndependent, Linear, Quadratic, Cosinus}; 
};

template< typename Mesh,
          TimeFunctionBase::TimeFunctionSet timefunction,
          typename Real,
          typename Index >
class TimeFunction
{
   
};

template< int Dim,
          typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
class TimeFunction< tnlGrid< Dim, MeshReal, Device, MeshIndex >,
                    TimeFunctionBase::TimeIndependent,
                    Real, Index >: public TimeFunctionBase
{
   public:
   enum setTimeFunction{ TimeIndependent };
   typedef tnlGrid< Dim, MeshReal, Device, MeshIndex > MeshType;
   typedef Real RealType;
   typedef typename MeshType::DeviceType DeviceType;
   typedef Index IndexType;
   typedef tnlSharedVector< RealType, DeviceType, IndexType > SharedVectorType;
   typedef tnlVector< RealType, DeviceType, IndexType> DofVectorType;
   
   RealType getTimeValue(const RealType& time) const;
   RealType getDerivation(const RealType& time) const;
   void applyInitTimeValues(SharedVectorType& u) const;
};

template< int Dim,
          typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
class TimeFunction< tnlGrid< Dim, MeshReal, Device, MeshIndex >,
                    TimeFunctionBase::Linear,
                    Real, Index >: public TimeFunctionBase
{
   public:
   enum setTimeFunction{ Linear };
   typedef tnlGrid< Dim, MeshReal, Device, MeshIndex > MeshType;
   typedef Real RealType;
   typedef typename MeshType::DeviceType DeviceType;
   typedef Index IndexType;
   typedef tnlSharedVector< RealType, DeviceType, IndexType > SharedVectorType;
   typedef tnlVector< RealType, DeviceType, IndexType> DofVectorType; 
   
   RealType getTimeValue(const RealType& time) const;
   RealType getDerivation(const RealType& time) const;
   void applyInitTimeValues(SharedVectorType& u) const;
};

template< int Dim,
          typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
class TimeFunction< tnlGrid< Dim, MeshReal, Device, MeshIndex >,
                    TimeFunctionBase::Quadratic,
                    Real, Index >: public TimeFunctionBase
{
   public:
   enum setTimeFunction{ Quadratic };
   typedef tnlGrid< Dim, MeshReal, Device, MeshIndex > MeshType;
   typedef Real RealType;
   typedef typename MeshType::DeviceType DeviceType;
   typedef Index IndexType;
   typedef tnlSharedVector< RealType, DeviceType, IndexType > SharedVectorType;
   typedef tnlVector< RealType, DeviceType, IndexType> DofVectorType;
   
   RealType getTimeValue(const RealType& time) const;
   RealType getDerivation(const RealType& time) const;
   void applyInitTimeValues(SharedVectorType& u) const;
};

template<>
template< int Dim,
          typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
class TimeFunction< tnlGrid< Dim, MeshReal, Device, MeshIndex >,
                    TimeFunctionBase::Cosinus,
                    Real, Index >: public TimeFunctionBase
{
   public:
   enum setTimeFunction{ Cosinus };
   typedef tnlGrid< Dim, MeshReal, Device, MeshIndex > MeshType;
   typedef Real RealType;
   typedef typename MeshType::DeviceType DeviceType;
   typedef Index IndexType;
   typedef tnlSharedVector< RealType, DeviceType, IndexType > SharedVectorType;
   typedef tnlVector< RealType, DeviceType, IndexType> DofVectorType;
   
   RealType getTimeValue(const RealType& time) const;
   RealType getDerivation(const RealType& time) const;
   void applyInitTimeValues(SharedVectorType& u) const;
};

#include "tnlTimeFunction_impl.h"

#endif	/* TNLTIMEFUNCTION_H */
