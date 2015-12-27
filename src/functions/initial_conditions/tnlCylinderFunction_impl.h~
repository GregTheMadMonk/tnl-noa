/***************************************************************************
                          tnlExpBumpFunction_impl.h  -  description
                             -------------------
    begin                : Dec 5, 2013
    copyright            : (C) 2013 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/

#ifndef TNLCYLINDERFUNCTION_IMPL_H_
#define TNLCYLINDERFUNCTION_IMPL_H_

#include <functions/initial_conditions/tnlCylinderFunction.h>

template< typename Real >
bool
tnlCylinderFunctionBase< Real >::
setup( const tnlParameterContainer& parameters,
       const tnlString& prefix )
{
   this->diameter = parameters.getParameter< double >( prefix + "diameter" );
   return true;
}

template< typename Real >
void tnlCylinderFunctionBase< Real >::setDiameter( const Real& sigma )
{
   this->diameter = diameter;
}

template< typename Real >
const Real& tnlCylinderFunctionBase< Real >::getDiameter() const
{
   return this->diameter;
}

/***
 * 1D
 */

template< typename Real >
tnlString
tnlCylinderFunction< 1, Real >::getType()
{
   return "tnlCylinderFunction< 1, " + ::getType< Real >() + tnlString( " >" );
}

template< typename Real >
tnlCylinderFunction< 1, Real >::tnlCylinderFunction()
{
}

template< typename Real >
   template< int XDiffOrder, 
             int YDiffOrder,
             int ZDiffOrder,
             typename Vertex >
#ifdef HAVE_CUDA
      __device__ __host__
#endif
Real
tnlCylinderFunction< 1, Real >::getValue( const Vertex& v,
                                         const Real& time ) const
{
   const RealType& x = v.x();
   if( YDiffOrder != 0 || ZDiffOrder != 0 )
      return 0.0;
   if( XDiffOrder == 0 )
      return ( ( x*x - this->diameter ) < 0 ) - ( ( x*x - this->diameter ) > 0 ) + 1;
   return 0.0;
}

/****
 * 2D
 */

template< typename Real >
tnlString
tnlCylinderFunction< 2, Real >::getType()
{
   return tnlString( "tnlCylinderFunction< 2, " ) + ::getType< Real >() + " >";
}

template< typename Real >
tnlCylinderFunction< 2, Real >::tnlCylinderFunction()
{
}

template< typename Real >
   template< int XDiffOrder,
             int YDiffOrder,
             int ZDiffOrder,
             typename Vertex >
#ifdef HAVE_CUDA
      __device__ __host__
#endif
Real
tnlCylinderFunction< 2, Real >::
getValue( const Vertex& v,
          const Real& time ) const
{
   const RealType& x = v.x();
   const RealType& y = v.y();
   if( ZDiffOrder != 0 )
      return 0.0;
   if( XDiffOrder == 0 && YDiffOrder == 0 )
      return ( ( x*x + y*y - this->diameter ) < 0 ) - ( ( x*x + y*y - this->diameter ) > 0 ) + 1;
   return 0.0;
}

/****
 * 3D
 */

template< typename Real >
tnlString
tnlCylinderFunction< 3, Real >::getType()
{
   return tnlString( "tnlCylinderFunction< 3, " ) + ::getType< Real >() + " >";
}

template< typename Real >
tnlCylinderFunction< 3, Real >::tnlCylinderFunction()
{
}

template< typename Real >
   template< int XDiffOrder,
             int YDiffOrder,
             int ZDiffOrder,
             typename Vertex >
#ifdef HAVE_CUDA
      __device__ __host__
#endif
Real
tnlCylinderFunction< 3, Real >::
getValue( const Vertex& v,
          const Real& time ) const
{
   const RealType& x = v.x();
   const RealType& y = v.y();
   const RealType& z = v.z();
   if( XDiffOrder == 0 && YDiffOrder == 0 && ZDiffOrder == 0 )
      return ( ( x*x + y*y + z*z - this->diameter ) < 0 ) - ( ( x*x + y*y + z*z - this->diameter ) > 0 ) + 1;
   return 0.0;
}


#endif /* TNLCYLINDERFUNCTION_IMPL_H_ */
