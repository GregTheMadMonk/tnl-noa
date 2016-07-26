/***************************************************************************
                          tnlHeatEquationEocRhs.h  -  description
                             -------------------
    begin                : Sep 8, 2014
    copyright            : (C) 2014 by oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/functions/tnlDomain.h>

namespace TNL {

template< typename ExactOperator,
          typename TestFunction,
          int Dimensions >
class tnlMeanCurvatureFlowEocRhs : public tnlDomain< Dimensions, SpaceDomain >
{
   public:

      typedef ExactOperator ExactOperatorType;
      typedef TestFunction TestFunctionType;
      typedef typename TestFunctionType::RealType RealType;
      typedef StaticVector< Dimensions, RealType > VertexType;

      bool setup( const Config::ParameterContainer& parameters,
                  const String& prefix = "" )
      {
         if( ! testFunction.setup( parameters, prefix ) )
            return false;
         return true;
      };

      template< typename Vertex,
                typename Real >
      __cuda_callable__
      Real operator()( const Vertex& vertex,
                       const Real& time ) const
      {
         return testFunction.getTimeDerivative( vertex, time )
                - exactOperator( testFunction, vertex, time );
      };

   protected:
 
      ExactOperator exactOperator;

      TestFunction testFunction;
};

} // namespace TNL
