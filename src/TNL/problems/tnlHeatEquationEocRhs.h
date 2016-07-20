/***************************************************************************
                          tnlHeatEquationEocRhs.h  -  description
                             -------------------
    begin                : Sep 8, 2014
    copyright            : (C) 2014 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

/***
 * Authors:
 * Oberhuber Tomas, tomas.oberhuber@fjfi.cvut.cz
 * Szekely Ondrej, ondra.szekely@gmail.com
 */

#pragma once

#include <TNL/functions/tnlDomain.h>

namespace TNL {

template< typename ExactOperator,
          typename TestFunction >
class tnlHeatEquationEocRhs
 : public tnlDomain< TestFunction::Dimensions, SpaceDomain >
{
   public:

      typedef ExactOperator ExactOperatorType;
      typedef TestFunction TestFunctionType;
      typedef typename TestFunction::RealType RealType;
      typedef typename TestFunction::VertexType VertexType;

      bool setup( const Config::ParameterContainer& parameters,
                  const String& prefix = "" )
      {
         if( ! testFunction.setup( parameters, prefix ) )
            return false;
         return true;
      }

      __cuda_callable__
      RealType operator()( const VertexType& vertex,
                         const RealType& time = 0.0 ) const
      {
         return testFunction.getTimeDerivative( vertex, time )
                - exactOperator( testFunction, vertex, time );
      }

   protected:
      ExactOperator exactOperator;

      TestFunction testFunction;
};

} // namespace TNL
