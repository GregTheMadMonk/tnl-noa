/***************************************************************************
                          HeatEquationEocRhs.h  -  description
                             -------------------
    begin                : Sep 8, 2014
    copyright            : (C) 2014 by oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/Functions/Domain.h>

namespace TNL {
namespace Problems {

template< typename ExactOperator,
          typename TestFunction,
          int Dimension >
class MeanCurvatureFlowEocRhs : public Domain< Dimension, SpaceDomain >
{
   public:

      typedef ExactOperator ExactOperatorType;
      typedef TestFunction TestFunctionType;
      typedef typename TestFunctionType::RealType RealType;
      typedef StaticVector< Dimension, RealType > PointType;

      bool setup( const Config::ParameterContainer& parameters,
                  const String& prefix = "" )
      {
         if( ! testFunction.setup( parameters, prefix ) )
            return false;
         return true;
      };

      template< typename Point,
                typename Real >
      __cuda_callable__
      Real operator()( const Point& vertex,
                       const Real& time ) const
      {
         return testFunction.getTimeDerivative( vertex, time )
                - exactOperator( testFunction, vertex, time );
      };

   protected:
 
      ExactOperator exactOperator;

      TestFunction testFunction;
};

} // namespace Problems
} // namespace TNL
