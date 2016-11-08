/***************************************************************************
                          FunctionInverseOperator.h  -  description
                             -------------------
    begin                : Feb 17, 2016
    copyright            : (C) 2016 by oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/String.h>
#include <TNL/Devices/Cuda.h>
#include <TNL/Operators/Operator.h>

namespace TNL {
namespace Operators {

template< typename OperatorT >
class FunctionInverseOperator
: public Operator< typename OperatorT::MeshType,
                      OperatorT::getDomainType(),
                      OperatorT::getPreimageEntitiesDimensions(),
                      OperatorT::getImageEntitiesDimensions(),
                      typename OperatorT::RealType,
                      typename OperatorT::IndexType >
{
   public:
 
      typedef OperatorT OperatorType;
      typedef typename OperatorType::RealType RealType;
      typedef typename OperatorType::IndexType IndexType;
      typedef FunctionInverseOperator< OperatorT > ThisType;
      typedef ThisType ExactOperatorType;
 
 
      FunctionInverseOperator( const OperatorType& operator_ )
      : operator_( operator_ ) {};
 
      static String getType()
      {
         return String( "FunctionInverseOperator< " ) + OperatorType::getType() + " >";
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

} // namespace Operators
} // namespace TNL

