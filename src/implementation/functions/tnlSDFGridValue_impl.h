/***************************************************************************
                          tnlSDFGridValue_impl.h  -  description
                             -------------------
    begin                : Oct 13, 2014
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

#ifndef TNLSDFGRIDVALUE_IMPL_H_
#define TNLSDFGRIDVALUE_IMPL_H_

#include <functions/tnlSDFGridValue.h>

template< typename Real >
tnlSDFGridValueBase< Real >::tnlSDFGridValueBase()
{
}

template< typename Mesh, typename Real >
bool tnlSDFGridValue< Mesh, 1, Real >::setup( const tnlParameterContainer& parameters,
        								 const tnlString& prefix)
{
	   this->mesh.load(parameters.GetParameter< tnlString >( "mesh" ));
	   this->u.load(parameters.GetParameter< tnlString >( "initial-condition" ));
	   this->origin=this->mesh.getOrigin();
	   this->hx=this->mesh.getHx();
   return true;
}
template< typename Mesh, typename Real >
bool tnlSDFGridValue< Mesh, 2, Real >::setup( const tnlParameterContainer& parameters,
        								 const tnlString& prefix)
{
	   this->mesh.load(parameters.GetParameter< tnlString >( "mesh" ));
	   this->u.load(parameters.GetParameter< tnlString >( "initial-condition" ));
	   this->dimensions=this->mesh.getDimensions();
	   this->origin=this->mesh.getOrigin();
	   this->hx=this->mesh.getHx();
	   this->hy=this->mesh.getHy();
   return true;
}
template< typename Mesh, typename Real >
bool tnlSDFGridValue< Mesh, 3, Real >::setup( const tnlParameterContainer& parameters,
        								 const tnlString& prefix)
{
	   this->mesh.load(parameters.GetParameter< tnlString >( "mesh" ));
	   this->u.load(parameters.GetParameter< tnlString >( "initial-condition" ));
	   this->dimensions=this->mesh.getDimensions();
	   this->origin=this->mesh.getOrigin();
	   this->hx=this->mesh.getHx();
	   this->hy=this->mesh.getHy();
	   this->hz=this->mesh.getHz();
   return true;
}

template< typename Mesh, typename Real >
void tnlSDFGridValue< Mesh, 1, Real >::bind( const Mesh& mesh,
        									  DofVectorType& dofVector )
{
	   const IndexType dofs = mesh.getNumberOfCells();
	   this->u.bind( dofVector.getData(), dofs );
	   this->origin=mesh.getOrigin();
	   this->hx=mesh.getHx();

}

template< typename Mesh, typename Real >
void tnlSDFGridValue< Mesh, 2, Real >::bind( const Mesh& mesh,
        									  DofVectorType& dofVector )
{
	   const IndexType dofs = mesh.getNumberOfCells();
	   this->u.bind( dofVector.getData(), dofs );
	   this->dimensions=mesh.getDimensions();
	   this->origin=mesh.getOrigin();
	   this->hx=mesh.getHx();
	   this->hy=mesh.getHy();
}

template< typename Mesh, typename Real >
void tnlSDFGridValue< Mesh, 3, Real >::bind( const Mesh& mesh,
        									  DofVectorType& dofVector )
{
	   const IndexType dofs = mesh.getNumberOfCells();
	   this->u.bind( dofVector.getData(), dofs );
	   this->dimensions=mesh.getDimensions();
	   this->origin=mesh.getOrigin();
	   this->hx=mesh.getHx();
	   this->hy=mesh.getHy();
	   this->hz=mesh.getHz();
}


template< typename Mesh, typename Real >
   template< int XDiffOrder,
             int YDiffOrder,
             int ZDiffOrder,
             typename Vertex >
Real tnlSDFGridValue< Mesh, 1, Real >::getValue( const Vertex& v,
        									const Real& time) const
{

   const Real& x =  (v.x() - this->origin.x()  ) / this->hx;

   if( XDiffOrder != 0 || YDiffOrder != 0 || ZDiffOrder != 0 )
      return 0.0;
   else
	   return u[x];
}


template< typename Mesh, typename Real >
   template< int XDiffOrder,
             int YDiffOrder,
             int ZDiffOrder,
             typename Vertex >
Real tnlSDFGridValue< Mesh, 2, Real >::getValue( const Vertex& v,
        									const Real& time) const
{

	   const IndexType& x = ((v.x() - this->origin.x()) / this->hx);
	   const IndexType& y = ((v.y() - this->origin.y()) / this->hy);

	   if( XDiffOrder != 0 || YDiffOrder != 0 || ZDiffOrder != 0 )
	      return 0.0;
	   else
		   return u[y * this->dimensions.x() + x];
}

template< typename Mesh, typename Real >
   template< int XDiffOrder,
             int YDiffOrder,
             int ZDiffOrder,
             typename Vertex >
Real tnlSDFGridValue< Mesh, 3, Real >::getValue( const Vertex& v,
        									const Real& time) const
{
	   const Real& x = floor( (v.x() - this->origin.x()) / this->hx);
	   const Real& y = floor( (v.y() - this->origin.y()) / this->hy);
	   const Real& z = floor( (v.z() - this->origin.z()) / this->hz);
	   if( XDiffOrder != 0 || YDiffOrder != 0 || ZDiffOrder != 0 )
	      return 0.0;
	   else
		   return u[( z * this->dimensions.y() + y ) * this->dimensions.x() + x];
}

#endif /* TNLSDFGRIDVALUE_IMPL_H_ */
