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

#include <functions/tnlDomain.h>

template< typename ExactOperator,
          typename TestFunction,
          int Dimensions >
class tnlMeanCurvatureFlowEocRhs : public tnlDomain< Dimensions, SpaceDomain >
{
   public:

      typedef ExactOperator ExactOperatorType;
      typedef TestFunction TestFunctionType;
      typedef typename TestFunctionType::RealType RealType;
      typedef tnlStaticVector< Dimensions, RealType > VertexType;

      bool setup( const tnlParameterContainer& parameters,
                  const tnlString& prefix = "" )
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


#endif /* TNLMEANCURVATUREFLOWEOCRHS_H_ */
