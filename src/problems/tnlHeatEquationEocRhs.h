/***************************************************************************
                          tnlHeatEquationEocRhs.h  -  description
                             -------------------
    begin                : Sep 8, 2014
    copyright            : (C) 2014 by Tomas Oberhuber et al.
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

/***
 * Authors:
 * Oberhuber Tomas, tomas.oberhuber@fjfi.cvut.cz
 * Szekely Ondrej, ondra.szekely@gmail.com
 */

#ifndef TNLHEATEQUATIONEOCRHS_H_
#define TNLHEATEQUATIONEOCRHS_H_

#include <functors/tnlFunction.h>

template< typename ExactOperator,
          typename TestFunction >
class tnlHeatEquationEocRhs
{
   public:

      typedef ExactOperator ExactOperatorType;
      typedef TestFunction TestFunctionType;

      //static constexpr tnlFunctionType getFunctionType() { return tnlAnalyticFunction; }     
      enum { functionType = tnlAnalyticFunction };
      
      bool setup( const tnlParameterContainer& parameters,
                  const tnlString& prefix = "" )
      {
         if( ! testFunction.setup( parameters, prefix ) )
            return false;
         return true;
      }

      template< typename Vertex,
                typename Real >
      __cuda_callable__
      Real getValue( const Vertex& vertex,
                     const Real& time ) const
      {
         return testFunction.getTimeDerivative( vertex, time )
                - exactOperator.getValue( testFunction, vertex, time );
      }

   protected:
      ExactOperator exactOperator;

      TestFunction testFunction;
};

/*
template< typename ExactOperator,
          typename TestFunction >
class tnlFunctionType< tnlHeatEquationEocRhs< ExactOperator, TestFunction > >
{
   public:

      enum { Type = tnlAnalyticFunction };
};
*/

#endif /* TNLHEATEQUATIONEOCRHS_H_ */
