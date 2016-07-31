/***************************************************************************
                          tnlFunctionInverseOperator.h  -  description
                             -------------------
    begin                : Feb 17, 2016
    copyright            : (C) 2016 by oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/String.h>
#include <TNL/Devices/Cuda.h>
#include <TNL/operators/tnlOperator.h>

namespace TNL {

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
 
      static String getType()
      {
         return String( "tnlFunctionInverseOperator< " ) + Operator::getType() + " >";
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

} // namespace TNL

