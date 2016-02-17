/***************************************************************************
                          tnlFunctionInverseOperator.h  -  description
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

#ifndef TNLFUNCTIONINVERSEOPERATOR_H
#define	TNLFUNCTIONINVERSEOPERATOR_H

#include <core/tnlString.h>
#include <core/tnlCuda.h>
#include <operators/tnlOperator.h>

template< typename Operator >
class tnlFunctionInverseOperator
: public tnlOperator< typename Operator::MeshType,
                       Operator::getDomainType(),
                       Operator::getPreimageEntitiesDimensions(),
                       Operator::getImageEntitiesDimensions(),
                       typename Operator::RealType,
                       typename Operator::IndexType >
{
   public:
      
      typedef Operator OperatorType;
      typedef typename OperatorType::RealType RealType;
      typedef typename OperatorType::IndexType IndexType;
      typedef tnlFunctionInverseOperator< Operator > ThisType;
      typedef ThisType ExactOperatorType;
      
      
      tnlFunctionInverseOperator( const OperatorType& operator_ )
      : operator_( operator_ ) {};
      
      static tnlString getType()
      {
         return tnlString( "tnlFunctionInverseOperator< " ) + Operator::getType() + " >";         
      }
      
      const OperatorType& getOperator() const { return this->operator_; }
      
      template< typename MeshFunction,
                typename MeshEntity >
      __cuda_callable__
      typename MeshFunction::RealType
      operator()( const MeshFunction& u,
                  const MeshEntity& entity,
                  const typename MeshFunction::RealType& time = 0.0 ) const
      {
         return 1.0 / operator_( u, entity, time );
      }
      
   protected:
      
      const OperatorType& operator_;
};


#endif	/* TNLINVERSEFUNCTIONOPERATOR_H */

