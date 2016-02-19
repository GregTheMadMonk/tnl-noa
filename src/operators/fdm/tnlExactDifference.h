/***************************************************************************
                          tnlExactDifference.h  -  description
                             -------------------
    begin                : Jan 10, 2016
    copyright            : (C) 2016 by Tomas Oberhuber
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

#ifndef TNLEXACTDIFFERENCE_H
#define	TNLEXACTDIFFERENCE_H

template< int Dimensions,
          int XDerivative,
          int YDerivative,
          int ZDerivative >
class tnlExactDifference
   : public tnlDomain< Dimensions, SpaceDomain >
{
   public:
      
      static tnlString getType()
      {
         return tnlString( "tnlExactDifference< " ) +
            tnlString( Dimensions ) + ", " +
            tnlString( XDerivative ) + ", " +
            tnlString( YDerivative ) + ", " +
            tnlString( ZDerivative ) + " >";
      }
      
      template< typename Function >
      typename Function::RealType operator()( 
         const Function& function,
         const typename Function::VertexType& vertex,
         const typename Function::RealType& time = 0 ) const
      {
         return function.template getPartialDerivative<
            XDerivative,
            YDerivative,
            ZDerivative >(
            vertex, 
            time );
      }
};


#endif	/* TNLEXACTDIFFERENCE_H */

