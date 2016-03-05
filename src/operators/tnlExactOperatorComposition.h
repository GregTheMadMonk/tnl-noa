/***************************************************************************
                          tnlExactOperatorComposition.h  -  description
                             -------------------
    begin                : Feb 17, 2016
    copyright            : (C) 2016 by oberhuber
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

#ifndef TNLEXACTOPERATORCOMPOSITION_H
#define TNLEXACTOPERATORCOMPOSITION_H

template< typename OuterOperator,
          typename InnerOperator >
class tnlExactOperatorComposition
{
   public:
      
      template< typename Function >
      __cuda_callable__ inline
      typename Function::RealType operator()( const Function& function,
                                              const typename Function::VertexType& v,
                                              const typename Function::RealType& time = 0.0 ) const
      {
         return OuterOperator( innerOperator( function, v, time), v, time );
      }
      
   protected:
      
      InnerOperator innerOperator;
      
      OuterOperator outerOperator;
};

#endif /* TNLEXACTOPERATORCOMPOSITION_H */

