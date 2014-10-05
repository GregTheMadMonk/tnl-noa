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

#ifndef TNLHEATEQUATIONEOCRHS_H_
#define TNLHEATEQUATIONEOCRHS_H_

template< typename ExactOperator,
          typename TestFunction >
class tnlHeatEquationEocRhs
{
   public:

      typedef ExactOperator ExactOperatorType;
      typedef TestFunction TestFunctionType;

      bool setup( const tnlParameterContainer& parameters )
      {
         if( ! testFunction.setup( parameters ) )
            return false;
         return true;
      };

      template< typename Vertex,
                typename Real >
      Real getValue( const Vertex& vertex,
                     const Real& time )
      {
         return testFunction.getTimeDerivative( vertex, time ) - exactOperator.getValue( testFunction, vertex, time );
      };

   protected:
      ExactOperator exactOperator;

      TestFunction testFunction;
};



#endif /* TNLHEATEQUATIONEOCRHS_H_ */
