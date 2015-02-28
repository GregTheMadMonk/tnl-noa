/***************************************************************************
                          tnlConstantFunction_impl.h  -  description
                             -------------------
    begin                : Aug 2, 2014
    copyright            : (C) 2014 by Tomas Oberhuber
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

#ifndef TNLCONSTANTFUNCTION_IMPL_H_
#define TNLCONSTANTFUNCTION_IMPL_H_

template< int Dimensions,
          typename Real >
tnlConstantFunction< Dimensions, Real >::
tnlConstantFunction()
: constant( 0.0 )
{
}

template< int Dimensions,
          typename Real >
void
tnlConstantFunction< Dimensions, Real >::
setConstant( const RealType& constant )
{
   this->constant = constant;
}

template< int Dimensions,
          typename Real >
const Real&
tnlConstantFunction< Dimensions, Real >::
getConstant() const
{
   return this->constant;
}

template< int FunctionDimensions,
          typename Real >
void
tnlConstantFunction< FunctionDimensions, Real >::
configSetup( tnlConfigDescription& config,
             const tnlString& prefix )
{
   config.addEntry     < double >( prefix + "constant", "Value of the constant function.", 0.0 );
}

template< int Dimensions,
          typename Real >
bool
tnlConstantFunction< Dimensions, Real >::
setup( const tnlParameterContainer& parameters,
       const tnlString& prefix )
{
   this->setConstant( parameters.getParameter< double >( prefix + "constant") );
   return true;
}

template< int Dimensions,
          typename Real >
   template< int XDiffOrder,
             int YDiffOrder,
             int ZDiffOrder,
             typename Vertex >
Real
tnlConstantFunction< Dimensions, Real >::
getValue( const Vertex& v,
          const Real& time ) const
{
   if( XDiffOrder || YDiffOrder || ZDiffOrder )
      return 0.0;
   return constant;
}

#endif