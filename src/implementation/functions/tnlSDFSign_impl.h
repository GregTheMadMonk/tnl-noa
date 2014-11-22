/***************************************************************************
                          tnlSDFSign_impl.h  -  description
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

#ifndef TNLSDFSIGN_IMPL_H_
#define TNLSDFSIGN_IMPL_H_

#include <functions/tnlSDFSign.h>

template< typename Real >
tnlSDFSignBase< Real >::tnlSDFSignBase()
: epsilon( 0 )
{
}

template< typename Mesh, typename Real, typename FunctionType, int SignMod >
bool tnlSDFSign< Mesh, 1, Real, FunctionType, SignMod >::setup( const tnlParameterContainer& parameters,
        								 const tnlString& prefix)
{
   this->epsilon = parameters.GetParameter< double >( "epsilon" );
   Mesh mesh;
   mesh.load(parameters.GetParameter< tnlString >( "mesh" ));
   this->epsilon*=mesh.getHx();
   this->function.setup(parameters, prefix);

   return true;
}
template< typename Mesh, typename Real, typename FunctionType, int SignMod >
bool tnlSDFSign< Mesh, 2, Real, FunctionType, SignMod >::setup( const tnlParameterContainer& parameters,
        								 const tnlString& prefix)
{
   this->epsilon = parameters.GetParameter< double >( "epsilon" );
   Mesh mesh;
   mesh.load(parameters.GetParameter< tnlString >( "mesh" ));
   this->epsilon=this->epsilon*sqrt(mesh.getHx()*mesh.getHx() + mesh.getHy()*mesh.getHy());
   this->function.setup(parameters, prefix);

   return true;
}
template< typename Mesh, typename Real, typename FunctionType, int SignMod >
bool tnlSDFSign< Mesh, 3, Real, FunctionType, SignMod >::setup( const tnlParameterContainer& parameters,
        								 const tnlString& prefix)
{
   this->epsilon = parameters.GetParameter< double >( "epsilon" );
   Mesh mesh;
   mesh.load(parameters.GetParameter< tnlString >( "mesh" ));
   this->epsilon*=sqrt(mesh.getHx()*mesh.getHx() + mesh.getHy()*mesh.getHy() + mesh.getHz()*mesh.getHz());
   this->function.setup(parameters, prefix);

   return true;
}


template< typename Real >
void tnlSDFSignBase< Real >::setEpsilon( const Real& epsilon )
{
   this->epsilon = epsilon;
}

template< typename Real >
Real tnlSDFSignBase< Real >::getEpsilon() const
{
   return this->epsilon;
}


template< typename Mesh, typename Real, typename FunctionType, int SignMod >
   template< int XDiffOrder,
             int YDiffOrder,
             int ZDiffOrder,
             typename Vertex >
Real tnlSDFSign< Mesh, 1, Real, FunctionType, SignMod >::getValue( const Vertex& v,
        									const Real& time) const
{
	   const Real& x = this->function.getValue(v, time);
	   if( XDiffOrder != 0 || YDiffOrder != 0 || ZDiffOrder != 0)
	      return 0.0;
	   else
	   {
			if(x > this->epsilon)
				return (Real)SignMod*1.0;
			else if (x < -this->epsilon)
				return (Real)SignMod*(-1.0);

			if(this->epsilon == 0.0)
				return 0.0;

			return (Real)SignMod*sin( ( M_PI * x ) / (2 * this->epsilon) );
	   }
}


template< typename Mesh, typename Real, typename FunctionType, int SignMod >
   template< int XDiffOrder,
             int YDiffOrder,
             int ZDiffOrder,
             typename Vertex >
Real tnlSDFSign< Mesh, 2, Real, FunctionType, SignMod >::getValue( const Vertex& v,
        									const Real& time) const
{

	   const Real& x = this->function.getValue(v, time);
	   if( XDiffOrder != 0 || YDiffOrder != 0 || ZDiffOrder != 0)
	      return 0.0;
	   else
	   {
			if(x > this->epsilon)
				return (Real)SignMod*1.0;
			else if (x < -this->epsilon)
				return (Real)SignMod*(-1.0);

			if(this->epsilon == 0.0)
				return 0.0;

			return (Real)SignMod*sin( ( M_PI * x ) / (2 * this->epsilon) );
	   }
}

template< typename Mesh, typename Real, typename FunctionType, int SignMod >
   template< int XDiffOrder,
             int YDiffOrder,
             int ZDiffOrder,
             typename Vertex >
Real tnlSDFSign< Mesh, 3, Real, FunctionType, SignMod >::getValue( const Vertex& v,
        									const Real& time) const
{
	   const Real& x = this->function.getValue(v, time);
	   if( XDiffOrder != 0 || YDiffOrder != 0 || ZDiffOrder != 0)
	      return 0.0;
	   else
	   {
			if(x > this->epsilon)
				return (Real)SignMod*1.0;
			else if (x < -this->epsilon)
				return (Real)SignMod*(-1.0);

			if(this->epsilon == 0.0)
				return 0.0;

			return (Real)SignMod*sin( ( M_PI * x ) / (2 * this->epsilon) );
	   }
}

#endif /* TNLSDFSIGN_IMPL_H_ */
