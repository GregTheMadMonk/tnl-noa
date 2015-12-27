/***************************************************************************
                          tnlHeatEquationEocRhs.h  -  description
                             -------------------
    begin                : Sep 8, 2014
    copyright            : (C) 2014 by oberhuber
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

#ifndef TNLMEANCURVATUREFLOWEOCRHS_H_
#define TNLMEANCURVATUREFLOWEOCRHS_H_

#include <functions/tnlFunction.h>

template< typename ExactOperator,
          typename TestFunction >
class tnlMeanCurvatureFlowEocRhs
{
   public:

      typedef ExactOperator ExactOperatorType;
      typedef TestFunction TestFunctionType;

      bool setup( const tnlParameterContainer& parameters,
                  const tnlString& prefix = "" )
      {
         if( ! testFunction.setup( parameters, prefix ) )
            return false;
         return true;
      };

      template< typename Vertex,
                typename Real >
#ifdef HAVE_CUDA
      __device__ __host__
#endif
      Real getValue( const Vertex& vertex,
                     const Real& time ) const
      {
         return testFunction.getTimeDerivative( vertex, time )
                - exactOperator.getValue( testFunction, vertex, time );
      };

   protected:
      ExactOperator exactOperator;

      TestFunction testFunction;
};

template< typename ExactOperator,
          typename TestFunction >
class tnlFunctionType< tnlMeanCurvatureFlowEocRhs< ExactOperator, TestFunction > >
{
   public:

      enum { Type = tnlAnalyticFunction };
};

#endif /* TNLMEANCURVATUREFLOWEOCRHS_H_ */
