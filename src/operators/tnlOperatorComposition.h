/***************************************************************************
                          tnlOperatorComposition.h  -  description
                             -------------------
    begin                : Jan 30, 2016
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

#ifndef TNLOPERATORCOMPOSITION_H
#define	TNLOPERATORCOMPOSITION_H

#include<functions/tnlOperatorFunction.h>
#include<functions/tnlMeshFunction.h>
#include<operators/tnlOperator.h>

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
class tnlOperatorComposition
   : public tnlOperator< typename InnerOperator::MeshType,
                         InnerOperator::getDomainType(),
                         InnerOperator::getPreimageEntitiesDimensions(),
                         OuterOperator::getImageEntitiesDimensions(),
                         typename InnerOperator::RealType,
                         typename OuterOperator::IndexType >   
{
      static_assert( is_same< typename OuterOperator::MeshType, typename InnerOperator::MeshType >::value,
         "Both operators have different mesh types." );
   public:
      
      typedef typename InnerOperator::MeshType MeshType;
      typedef tnlMeshFunction< MeshType, InnerOperator::getPreimageEntitiesDimensions() > PreimageFunctionType;
      typedef tnlMeshFunction< MeshType, InnerOperator::getImageEntitiesDimensions() > ImageFunctionType;
      typedef tnlOperatorFunction< InnerOperator, PreimageFunctionType, InnerBoundaryConditions > InnerOperatorFunction;
      typedef tnlOperatorFunction< InnerOperator, ImageFunctionType > OuterOperatorFunction;
      typedef typename InnerOperator::RealType RealType;
      typedef typename InnerOperator::IndexType IndexType;
      
      static constexpr int getPreimageEntitiesDimensions() { return InnerOperator::getImageEntitiesDimensions(); };
      static constexpr int getImageEntitiesDimensions() { return OuterOperator::getImageEntitiesDimensions(); };
      
      tnlOperatorComposition( OuterOperator& outerOperator,
                              InnerOperator& innerOperator,
                              const InnerBoundaryConditions& innerBoundaryConditions,
                              const MeshType& mesh )
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
         static_assert( MeshFunction::getDimensions() == InnerOperator::getDimensions(),
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
class tnlOperatorComposition< OuterOperator, InnerOperator, void >
   : public tnlDomain< InnerOperator::getDimensions(), InnerOperator::getDomainType() >   
{
      static_assert( is_same< typename OuterOperator::MeshType, typename InnerOperator::MeshType >::value,
         "Both operators have different mesh types." );
   public:
      
      typedef typename InnerOperator::MeshType MeshType;
      typedef tnlMeshFunction< MeshType, InnerOperator::getPreimageEntitiesDimensions() > PreimageFunctionType;
      typedef tnlMeshFunction< MeshType, InnerOperator::getImageEntitiesDimensions() > ImageFunctionType;
      typedef tnlOperatorFunction< InnerOperator, PreimageFunctionType, void > InnerOperatorFunction;
      typedef tnlOperatorFunction< InnerOperator, ImageFunctionType > OuterOperatorFunction;
      typedef typename InnerOperator::RealType RealType;
      typedef typename InnerOperator::IndexType IndexType;
      
      tnlOperatorComposition( const OuterOperator& outerOperator,
                              InnerOperator& innerOperator,
                              const MeshType& mesh )
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
         /*tnlMeshFunction< MeshType, MeshType::getMeshDimensions() - 1 > f( this->innerOperatorFunction.getMesh() );
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
         static_assert( MeshFunction::getDimensions() == InnerOperator::getDimensions(),
            "Mesh function and operator have both different number of dimensions." );
         //InnerOperatorFunction innerOperatorFunction( innerOperator, function );
         return outerOperator( innerOperatorFunction, meshEntity, time );
      }      
   
   protected:
      
      const OuterOperator& outerOperator;
      
      InnerOperatorFunction innerOperatorFunction;      
};


#endif	/* TNLOPERATORCOMPOSITION_H */

