/***************************************************************************
                          ConstantFunction_impl.h  -  description
                             -------------------
    begin                : Aug 2, 2014
    copyright            : (C) 2014 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

namespace TNL {
namespace Functions {
namespace Analytic {   

template< int Dimensions,
          typename Real >
ConstantFunction< Dimensions, Real >::
ConstantFunction()
: constant( 0.0 )
{
}

template< int Dimensions,
          typename Real >
void
ConstantFunction< Dimensions, Real >::
setConstant( const RealType& constant )
{
   this->constant = constant;
}

template< int Dimensions,
          typename Real >
const Real&
ConstantFunction< Dimensions, Real >::
getConstant() const
{
   return this->constant;
}

template< int FunctionDimensions,
          typename Real >
void
ConstantFunction< FunctionDimensions, Real >::
configSetup( Config::ConfigDescription& config,
             const String& prefix )
{
   config.addEntry     < double >( prefix + "constant", "Value of the constant function.", 0.0 );
}

template< int Dimensions,
          typename Real >
bool
ConstantFunction< Dimensions, Real >::
setup( const Config::ParameterContainer& parameters,
       const String& prefix )
{
   this->setConstant( parameters.getParameter< double >( prefix + "constant") );
   return true;
}

template< int Dimensions,
          typename Real >
   template< int XDiffOrder,
             int YDiffOrder,
             int ZDiffOrder >
Real
ConstantFunction< Dimensions, Real >::
getPartialDerivative( const VertexType& v,
                      const Real& time ) const
{
   if( XDiffOrder || YDiffOrder || ZDiffOrder )
      return 0.0;
   return constant;
}

} // namespace Analytic
} // namespace Functions
} // namespace TNL
