/***************************************************************************
                          OperatorComposition.h  -  description
                             -------------------
    begin                : Jan 30, 2016
    copyright            : (C) 2016 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/Functions/OperatorFunction.h>
#include <TNL/Functions/MeshFunction.h>
#include <TNL/Operators/Operator.h>
#include <TNL/Operators/ExactOperatorComposition.h>

namespace TNL {
namespace Operators {

/****
 * This object serves for composition of two operators F and G into an operator F( G( u ) ).
 * The function u must be set in the constructor or by a method setPreimageFunction.
 * Each time the function u is changed, the method refresh() or deepRefresh() must be called
 * before using the operator(). The function u which is passed to the operator() is,in fact,
 * omitted in this case.
 */

template< typename OuterOperator,
          typename InnerOperator,
          typename InnerBoundaryConditions = void >
class OperatorComposition
   : public Operator< typename InnerOperator::MeshType,
                         InnerOperator::getDomainType(),
                         InnerOperator::getPreimageEntitiesDimension(),
                         OuterOperator::getImageEntitiesDimension(),
                         typename InnerOperator::RealType,
                         typename OuterOperator::IndexType >
{
      static_assert( std::is_same< typename OuterOperator::MeshType, typename InnerOperator::MeshType >::value,
         "Both operators have different mesh types." );
   public:
 
      typedef typename InnerOperator::MeshType MeshType;
      typedef Functions::MeshFunction< MeshType, InnerOperator::getPreimageEntitiesDimension() > PreimageFunctionType;
      typedef Functions::MeshFunction< MeshType, InnerOperator::getImageEntitiesDimension() > ImageFunctionType;
      typedef Functions::OperatorFunction< InnerOperator, PreimageFunctionType, InnerBoundaryConditions > InnerOperatorFunction;
      typedef Functions::OperatorFunction< InnerOperator, ImageFunctionType > OuterOperatorFunction;
      typedef typename InnerOperator::RealType RealType;
      typedef typename InnerOperator::IndexType IndexType;
      typedef ExactOperatorComposition< typename OuterOperator::ExactOperatorType,
                                           typename InnerOperator::ExactOperatorType > ExactOperatorType;
      typedef SharedPointer< MeshType > MeshPointer;
      
      static constexpr int getPreimageEntitiesDimension() { return InnerOperator::getImageEntitiesDimension(); };
      static constexpr int getImageEntitiesDimension() { return OuterOperator::getImageEntitiesDimension(); };
 
      OperatorComposition( OuterOperator& outerOperator,
                              InnerOperator& innerOperator,
                              const InnerBoundaryConditions& innerBoundaryConditions,
                              const MeshPointer& mesh )
      : outerOperator( outerOperator ),
        innerOperatorFunction( innerOperator, innerBoundaryConditions, mesh ){};
 
      void setPreimageFunction( const PreimageFunctionType& preimageFunction )
      {
         this->innerOperatorFunction.setPreimageFunction( preimageFunction );
      }
 
      PreimageFunctionType& getPreimageFunction()
      {
         return this->innerOperatorFunction.getPreimageFunction();
      }

      const PreimageFunctionType& getPreimageFunction() const
      {
         return this->innerOperatorFunction.getPreimageFunction();
      }
 
      InnerOperator& getInnerOperator() { return this->innerOperatorFunction.getOperator(); }
 
      const InnerOperator& getInnerOperator() const { return this->innerOperatorFunction.getOperator(); }
 
      OuterOperator& getOuterOperator() { return this->outerOperator(); };
 
      const OuterOperator& getOuterOperator() const { return this->outerOperator(); };
 
      bool refresh( const RealType& time = 0.0 )
      {
         return this->innerOperatorFunction.refresh( time );
      }
 
      bool deepRefresh( const RealType& time = 0.0 )
      {
         return this->innerOperatorFunction.deepRefresh( time );
      }
 
      template< typename MeshFunction, typename MeshEntity >
      __cuda_callable__
      RealType operator()(
         const MeshFunction& function,
         const MeshEntity& meshEntity,
         const RealType& time = 0.0 ) const
      {
         static_assert( MeshFunction::getMeshDimension() == InnerOperator::getDimension(),
            "Mesh function and operator have both different number of dimensions." );
         //InnerOperatorFunction innerOperatorFunction( innerOperator, function );
         return outerOperator( innerOperatorFunction, meshEntity, time );
      }
 
   protected:
 
      OuterOperator& outerOperator;
 
      InnerOperatorFunction innerOperatorFunction;
};

template< typename OuterOperator,
          typename InnerOperator >
class OperatorComposition< OuterOperator, InnerOperator, void >
   : public Functions::Domain< InnerOperator::getDimension(), InnerOperator::getDomainType() >
{
      static_assert( std::is_same< typename OuterOperator::MeshType, typename InnerOperator::MeshType >::value,
         "Both operators have different mesh types." );
   public:
 
      typedef typename InnerOperator::MeshType MeshType;
      typedef Functions::MeshFunction< MeshType, InnerOperator::getPreimageEntitiesDimension() > PreimageFunctionType;
      typedef Functions::MeshFunction< MeshType, InnerOperator::getImageEntitiesDimension() > ImageFunctionType;
      typedef Functions::OperatorFunction< InnerOperator, PreimageFunctionType, void > InnerOperatorFunction;
      typedef Functions::OperatorFunction< InnerOperator, ImageFunctionType > OuterOperatorFunction;
      typedef typename InnerOperator::RealType RealType;
      typedef typename InnerOperator::IndexType IndexType;
      typedef SharedPointer< MeshType > MeshPointer;
      
      OperatorComposition( const OuterOperator& outerOperator,
                              InnerOperator& innerOperator,
                              const MeshPointer& mesh )
      : outerOperator( outerOperator ),
        innerOperatorFunction( innerOperator, mesh ){};
 
      void setPreimageFunction( PreimageFunctionType& preimageFunction )
      {
         this->innerOperatorFunction.setPreimageFunction( preimageFunction );
      }
 
      PreimageFunctionType& getPreimageFunction()
      {
         return this->innerOperatorFunction.getPreimageFunction();
      }

      const PreimageFunctionType& getPreimageFunction() const
      {
         return this->innerOperatorFunction.getPreimageFunction();
      }
 
      bool refresh( const RealType& time = 0.0 )
      {
         return this->innerOperatorFunction.refresh( time );
         /*MeshFunction< MeshType, MeshType::getMeshDimension() - 1 > f( this->innerOperatorFunction.getMesh() );
         f = this->innerOperatorFunction;
         this->innerOperatorFunction.getPreimageFunction().write( "preimageFunction", "gnuplot" );
         f.write( "innerFunction", "gnuplot" );
         return true;*/
      }
 
      bool deepRefresh( const RealType& time = 0.0 )
      {
         return this->innerOperatorFunction.deepRefresh( time );
         /*this->innerOperatorFunction.write( "innerFunction", "gnuplot" );
          return true;*/
      }
 
      template< typename MeshFunction, typename MeshEntity >
      __cuda_callable__
      RealType operator()(
         const MeshFunction& function,
         const MeshEntity& meshEntity,
         const RealType& time = 0.0 ) const
      {
         static_assert( MeshFunction::getMeshDimension() == InnerOperator::getDimension(),
            "Mesh function and operator have both different number of dimensions." );
         //InnerOperatorFunction innerOperatorFunction( innerOperator, function );
         return outerOperator( innerOperatorFunction, meshEntity, time );
      }
 
   protected:
 
      const OuterOperator& outerOperator;
 
      InnerOperatorFunction innerOperatorFunction;
};

} // namespace Operators
} // namespace TNL

