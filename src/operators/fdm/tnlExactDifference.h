/***************************************************************************
                          tnlExactDifference.h  -  description
                             -------------------
    begin                : Jan 10, 2016
    copyright            : (C) 2016 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

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
      __cuda_callable__
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

