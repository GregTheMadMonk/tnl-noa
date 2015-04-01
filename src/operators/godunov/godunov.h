/***************************************************************************
                          godunov.h  -  description
                             -------------------
    begin                : Jul 8 , 2014
    copyright            : (C) 2014 by Tomas Sobotik
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/

#ifndef GODUNOV_H_
#define GODUNOV_H_

#include <matrices/tnlCSRMatrix.h>
#include <solvers/preconditioners/tnlDummyPreconditioner.h>
#include <solvers/tnlSolverMonitor.h>
#include <core/tnlLogger.h>
#include <core/vectors/tnlVector.h>
#include <core/vectors/tnlSharedVector.h>
#include <core/mfilename.h>
#include <mesh/tnlGrid.h>


template< typename Mesh,
		  typename Real,
		  typename Index,
		  typename Function >
class godunovScheme
{
};




template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index,
		  typename Function >
class godunovScheme< tnlGrid< 1,MeshReal, Device, MeshIndex >, Real, Index, Function >
{
public:
	typedef Real RealType;
	typedef Device DeviceType;
	typedef Index IndexType;
	typedef tnlGrid< 1, Real, Device, Index > MeshType;
	typedef tnlVector< RealType, DeviceType, IndexType> DofVectorType;
	typedef typename MeshType::CoordinatesType CoordinatesType;



	static tnlString getType();

	RealType positivePart(const RealType arg) const;

	RealType negativePart(const RealType arg) const;

	RealType sign(const RealType x, const RealType eps) const;

    template< typename Vector >
 #ifdef HAVE_CUDA
    __device__ __host__
 #endif
    Real getValue( const MeshType& mesh,
                   const IndexType cellIndex,
                   const CoordinatesType& coordinates,
                   const Vector& u,
                   const RealType& time ) const;


	bool init( const tnlParameterContainer& parameters );


protected:

	Function f;

	MeshType originalMesh;

	DofVectorType dofVector;

	RealType h;

	RealType epsilon;


};






template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index,
		  typename Function >
class godunovScheme< tnlGrid< 2,MeshReal, Device, MeshIndex >, Real, Index, Function >
{
public:
	typedef Real RealType;
	typedef Device DeviceType;
	typedef Index IndexType;
	typedef tnlGrid< 2, Real, Device, Index > MeshType;
	typedef tnlVector< RealType, DeviceType, IndexType> DofVectorType;
	typedef typename MeshType::CoordinatesType CoordinatesType;


	static tnlString getType();

    RealType positivePart(const RealType arg) const;

    RealType negativePart(const RealType arg) const;

    RealType sign(const RealType x, const Real eps) const;

    template< typename Vector >
 #ifdef HAVE_CUDA
    __device__ __host__
 #endif
    Real getValue( const MeshType& mesh,
                   const IndexType cellIndex,
                   const CoordinatesType& coordinates,
                   const Vector& u,
                   const RealType& time ) const;


    bool init( const tnlParameterContainer& parameters );


protected:

    Function f;

 	MeshType originalMesh;

    DofVectorType dofVector;

    RealType hx;
    RealType hy;

    RealType epsilon;


};


template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index,
		  typename Function >
class godunovScheme< tnlGrid< 3,MeshReal, Device, MeshIndex >, Real, Index, Function >
{

public:
	typedef Real RealType;
	typedef Device DeviceType;
	typedef Index IndexType;
	typedef tnlGrid< 3, Real, Device, Index > MeshType;
	typedef tnlVector< RealType, DeviceType, IndexType> DofVectorType;
	typedef typename MeshType::CoordinatesType CoordinatesType;



	static tnlString getType();

    RealType positivePart(const RealType arg) const;

    RealType negativePart(const RealType arg) const;

    RealType sign(const RealType x, const Real eps) const;

    template< typename Vector >
 #ifdef HAVE_CUDA
    __device__ __host__
 #endif
    Real getValue( const MeshType& mesh,
                   const IndexType cellIndex,
                   const CoordinatesType& coordinates,
                   const Vector& u,
                   const RealType& time ) const;


    bool init( const tnlParameterContainer& parameters );


protected:

    Function f;

 	MeshType originalMesh;

    DofVectorType dofVector;

    RealType hx;
    RealType hy;
    RealType hz;

    RealType epsilon;

};



#include <operators/godunov/godunov1D_impl.h>
#include <operators/godunov/godunov2D_impl.h>
#include <operators/godunov/godunov3D_impl.h>


#endif /* GODUNOV_H_ */
