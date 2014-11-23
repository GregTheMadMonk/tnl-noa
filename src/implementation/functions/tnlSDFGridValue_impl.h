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
   return true;
}
template< typename Mesh, typename Real >
bool tnlSDFGridValue< Mesh, 2, Real >::setup( const tnlParameterContainer& parameters,
        								 const tnlString& prefix)
{
   this->mesh.load(parameters.GetParameter< tnlString >( "mesh" ));
   this->u.load(parameters.GetParameter< tnlString >( "initial-condition" ));
   return true;
}
template< typename Mesh, typename Real >
bool tnlSDFGridValue< Mesh, 3, Real >::setup( const tnlParameterContainer& parameters,
        								 const tnlString& prefix)
{
   this->mesh.load(parameters.GetParameter< tnlString >( "mesh" ));
   this->u.load(parameters.GetParameter< tnlString >( "initial-condition" ));
   return true;
}



template< typename Mesh, typename Real >
   template< int XDiffOrder,
             int YDiffOrder,
             int ZDiffOrder,
             typename Vertex >
Real tnlSDFGridValue< Mesh, 1, Real >::getValue( const Vertex& v,
        									const Real& time) const
{

   const Real& x =  (v.x() - this->mesh.getOrigin().x()  ) / this->mesh.getHx();

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

	   const IndexType& x = ((v.x() - this->mesh.getOrigin().x()) / this->mesh.getHx());
	   const IndexType& y = ((v.y() - this->mesh.getOrigin().y()) / this->mesh.getHy());

	   if( XDiffOrder != 0 || YDiffOrder != 0 || ZDiffOrder != 0 )
	      return 0.0;
	   else
		   return u[y * this->mesh.getDimensions().x() + x];
}

template< typename Mesh, typename Real >
   template< int XDiffOrder,
             int YDiffOrder,
             int ZDiffOrder,
             typename Vertex >
Real tnlSDFGridValue< Mesh, 3, Real >::getValue( const Vertex& v,
        									const Real& time) const
{
	   const Real& x = floor( (v.x() - this->mesh.getOrigin().x()) / this->mesh.getHx());
	   const Real& y = floor( (v.y() - this->mesh.getOrigin().y()) / this->mesh.getHy());
	   const Real& z = floor( (v.z() - this->mesh.getOrigin().z()) / this->mesh.getHz());
	   if( XDiffOrder != 0 || YDiffOrder != 0 || ZDiffOrder != 0 )
	      return 0.0;
	   else
		   return u[( z * this->mesh.getDimensions().y() + y ) * this->mesh.getDimensions().x() + x];
}

#endif /* TNLSDFGRIDVALUE_IMPL_H_ */
