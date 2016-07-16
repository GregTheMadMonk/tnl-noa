/***************************************************************************
                          tnlExactOperatorFunction.h  -  description
                             -------------------
    begin                : Jan 5, 2016
    copyright            : (C) 2016 by oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#ifndef TNLEXACTOPERATORFUNCTION_H
#define	TNLEXACTOPERATORFUNCTION_H

#include <functions/tnlDomain.h>

template< typename Operator,
          typename Function >
class tnlExactOperatorFunction : public tnlDomain< Operator::getDimensions(), SpaceDomain >
{
   static_assert( Operator::getDimensions() == Function::getDimensions(),
      "Operator and function have different number of domain dimensions." );
 
   public:
 
      typedef Operator OperatorType;
      typedef Function FunctionType;
      typedef typename FunctionType::RealType RealType;
      typedef typename FunctionType::VertexType VertexType;
 
      static constexpr int getDimensions(){ return Operator::getDimensions(); };
 
      tnlExactOperatorFunction(
         const OperatorType& operator_,
         const FunctionType& function )
      : operator_( operator_ ), function( function ) {};
 
      __cuda_callable__
      RealType operator()(
         const VertexType& vertex,
         const RealType& time ) const
      {
         return this->operator_( function, vertex, time );
      }
 
   protected:
 
      const OperatorType& operator_;
 
      const FunctionType& function;
};

#endif	/* TNLEXACTOPERATORFUNCTION_H */

