#ifndef TNLTIMEFUNCTION_IMPL_H
#define	TNLTIMEFUNCTION_IMPL_H

#include "tnlTimeFunction.h"

template< int Dim,
          typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
Real
TimeFunction< tnlGrid< Dim, MeshReal, Device, MeshIndex >,
              TimeFunctionBase::TimeIndependent,
              Real, Index >::
getTimeValue( const RealType& time ) const
{
   return 1.0;
}

template< int Dim,
          typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
Real
TimeFunction< tnlGrid< Dim, MeshReal, Device, MeshIndex >,
              TimeFunctionBase::Linear,
              Real, Index >::
getTimeValue( const RealType& time ) const
{
   return time;
}

template< int Dim,
          typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
Real
TimeFunction< tnlGrid< Dim, MeshReal, Device, MeshIndex >,
              TimeFunctionBase::Quadratic,
              Real, Index >::
getTimeValue( const RealType& time ) const
{
   return time*time;
}

template< int Dim,
          typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
Real
TimeFunction< tnlGrid< Dim, MeshReal, Device, MeshIndex >,
              TimeFunctionBase::Cosinus,
              Real, Index >::
getTimeValue( const RealType& time ) const
{
   return cos( time );
}

template< int Dim,
          typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
Real
TimeFunction< tnlGrid< Dim, MeshReal, Device, MeshIndex >,
              TimeFunctionBase::TimeIndependent,
              Real, Index >::
getDerivation( const RealType& time ) const
{
   return 0.0;
}

template< int Dim,
          typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
Real
TimeFunction< tnlGrid< Dim, MeshReal, Device, MeshIndex >,
              TimeFunctionBase::Linear,
              Real, Index >::
getDerivation( const RealType& time ) const
{
   return 1.0;
}

template< int Dim,
          typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
Real
TimeFunction< tnlGrid< Dim, MeshReal, Device, MeshIndex >,
              TimeFunctionBase::Quadratic,
              Real, Index >::
getDerivation( const RealType& time ) const
{
   return 2.0*time;
}

template< int Dim,
          typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
Real
TimeFunction< tnlGrid< Dim, MeshReal, Device, MeshIndex >,
              TimeFunctionBase::Cosinus,
              Real, Index >::
getDerivation( const RealType& time ) const
{
   return -sin(time);
}

template< int Dim,
          typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
void
TimeFunction< tnlGrid< Dim, MeshReal, Device, MeshIndex >,
              TimeFunctionBase::TimeIndependent,
              Real, Index >::
applyInitTimeValues( SharedVectorType& u ) const
{
   
}

template< int Dim,
          typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
void
TimeFunction< tnlGrid< Dim, MeshReal, Device, MeshIndex >,
              TimeFunctionBase::Linear,
              Real, Index >::
applyInitTimeValues( SharedVectorType& u ) const
{
   #ifdef HAVE_OPENMP
      #pragma omp parallel for
   #endif
   for(IndexType i=1; i<(u.getSize()-1); i++)
   {         
     u[i] = 0;
   }
}

template< int Dim,
          typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
void
TimeFunction< tnlGrid< Dim, MeshReal, Device, MeshIndex >,
              TimeFunctionBase::Quadratic,
              Real, Index >::
applyInitTimeValues( SharedVectorType& u ) const
{
   #ifdef HAVE_OPENMP
      #pragma omp parallel for
   #endif
   for(IndexType i=1; i<(u.getSize()-1); i++)
   {         
     u[i] = 0;
   }
}

template< int Dim,
          typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
void
TimeFunction< tnlGrid< Dim, MeshReal, Device, MeshIndex >,
              TimeFunctionBase::Cosinus,
              Real, Index >::
applyInitTimeValues( SharedVectorType& u ) const
{

}

#endif	/* TNLTIMEFUNCTION_IMPL_H */
