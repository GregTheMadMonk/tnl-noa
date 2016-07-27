/***************************************************************************
                          tnlOperatorFunction.h  -  description
                             -------------------
    begin                : Dec 31, 2015
    copyright            : (C) 2015 by oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <type_traits>
#include <TNL/Devices/Cuda.h>
#include <TNL/functions/tnlMeshFunction.h>
#include <TNL/solvers/pde/tnlBoundaryConditionsSetter.h>

namespace TNL {

/***
 * This class evaluates given operator on given preimageFunction. If the flag
 * EvaluateOnFly is set on true, the values on particular mesh entities
 * are computed just  when operator() is called. If the EvaluateOnFly flag
 * is 'false', values on all mesh entities are evaluated by calling a method
 * refresh() they are stores in internal mesh preimageFunction and the operator()
 * just returns precomputed values. If BoundaryConditions are void then the
 * values on the boundary mesh entities are undefined. In this case, the mesh
 * preimageFunction evaluator evaluates this preimageFunction only on the INTERIOR mesh entities.
 */
   
template< typename Operator,
          typename MeshFunction,
          typename BoundaryConditions = void,
          bool EvaluateOnFly = false >
class tnlOperatorFunction{};

/****
 * Specialization for 'On the fly' evaluation with the boundary conditions does not make sense.
 */
template< typename Operator,
          typename MeshFunction,
          typename BoundaryConditions >
class tnlOperatorFunction< Operator, MeshFunction, BoundaryConditions, true >
 : public tnlDomain< Operator::getMeshDimensions(), MeshDomain >
{
};

/****
 * Specialization for 'On the fly' evaluation and no boundary conditions.
 */
template< typename Operator,
          typename MeshFunction >
class tnlOperatorFunction< Operator, MeshFunction, void, true >
 : public tnlDomain< Operator::getDimensions(), Operator::getDomainType() >
{
   public:
 
      static_assert( MeshFunction::getDomainType() == MeshDomain ||
                     MeshFunction::getDomainType() == MeshInteriorDomain ||
                     MeshFunction::getDomainType() == MeshBoundaryDomain,
         "Only mesh preimageFunctions may be used in the operator preimageFunction. Use tnlExactOperatorFunction instead of tnlOperatorFunction." );
      static_assert( std::is_same< typename Operator::MeshType, typename MeshFunction::MeshType >::value,
          "Both, operator and mesh preimageFunction must be defined on the same mesh." );
 
      typedef Operator OperatorType;
      typedef MeshFunction FunctionType;
      typedef typename OperatorType::MeshType MeshType;
      typedef typename OperatorType::RealType RealType;
      typedef typename OperatorType::DeviceType DeviceType;
      typedef typename OperatorType::IndexType IndexType;
      typedef typename OperatorType::ExactOperatorType ExactOperatorType;
      typedef tnlMeshFunction< MeshType, OperatorType::getPreimageEntitiesDimensions() > PreimageFunctionType;
 
      static constexpr int getEntitiesDimensions() { return OperatorType::getImageEntitiesDimensions(); };
 
      tnlOperatorFunction( const OperatorType& operator_ )
      :  operator_( operator_ ), preimageFunction( 0 ){};
 
      tnlOperatorFunction( const OperatorType& operator_,
                           const FunctionType& preimageFunction )
      :  operator_( operator_ ), preimageFunction( &preimageFunction ){};
 
      const MeshType& getMesh() const
      {
         Assert( this->preimageFunction, std::cerr << "The preimage function was not set." << std::endl );
         return this->preimageFunction->getMesh();
      };
 
      void setPreimageFunction( const FunctionType& preimageFunction ) { this->preimageFunction = &preimageFunction; }
 
      Operator& getOperator() { return this->operator_; }
 
      const Operator& getOperator() const { return this->operator_; }
 
      bool refresh( const RealType& time = 0.0 ) { return true; };
 
      bool deepRefresh( const RealType& time = 0.0 ) { return true; };
 
      template< typename MeshEntity >
      __cuda_callable__
      RealType operator()(
         const MeshEntity& meshEntity,
         const RealType& time = 0.0 ) const
      {
         Assert( this->preimageFunction, std::cerr << "The preimage function was not set." << std::endl );
         return operator_( *preimageFunction, meshEntity, time );
      }
 
   protected:
 
      const Operator& operator_;
 
      const FunctionType* preimageFunction;
 
      template< typename, typename > friend class tnlMeshFunctionEvaluator;
};

/****
 * Specialization for precomputed evaluation and no boundary conditions.
 */
template< typename Operator,
          typename PreimageFunction >
class tnlOperatorFunction< Operator, PreimageFunction, void, false >
 : public tnlDomain< Operator::getDimensions(), Operator::getDomainType() >
{
   public:
 
      static_assert( PreimageFunction::getDomainType() == MeshDomain ||
                     PreimageFunction::getDomainType() == MeshInteriorDomain ||
                     PreimageFunction::getDomainType() == MeshBoundaryDomain,
         "Only mesh preimageFunctions may be used in the operator preimageFunction. Use tnlExactOperatorFunction instead of tnlOperatorFunction." );
      static_assert( std::is_same< typename Operator::MeshType, typename PreimageFunction::MeshType >::value,
          "Both, operator and mesh preimageFunction must be defined on the same mesh." );
 
      typedef Operator OperatorType;
      typedef typename OperatorType::MeshType MeshType;
      typedef typename OperatorType::RealType RealType;
      typedef typename OperatorType::DeviceType DeviceType;
      typedef typename OperatorType::IndexType IndexType;
      typedef PreimageFunction PreimageFunctionType;
      typedef tnlMeshFunction< MeshType, Operator::getImageEntitiesDimensions() > ImageFunctionType;
      typedef tnlOperatorFunction< Operator, PreimageFunction, void, true > OperatorFunction;
      typedef typename OperatorType::ExactOperatorType ExactOperatorType;
 
      static constexpr int getEntitiesDimensions() { return OperatorType::getImageEntitiesDimensions(); };
 
      tnlOperatorFunction( OperatorType& operator_,
                           const MeshType& mesh )
      :  operator_( operator_ ), imageFunction( mesh )
      {};
 
      tnlOperatorFunction( OperatorType& operator_,
                           PreimageFunctionType& preimageFunction )
      :  operator_( operator_ ), imageFunction( preimageFunction.getMesh() ), preimageFunction( &preimageFunction )
      {};
 
      const MeshType& getMesh() const { return this->imageFunction.getMesh(); };
 
      ImageFunctionType& getImageFunction() { return this->imageFunction; };
 
      const ImageFunctionType& getImageFunction() const { return this->imageFunction; };
 
      void setPreimageFunction( PreimageFunction& preimageFunction )
      {
         this->preimageFunction = &preimageFunction;
         this->imageFunction.setMesh( preimageFunction.getMesh() );
      };
 
      const PreimageFunctionType& getPreimageFunction() const { return *this->preimageFunction; };
 
      Operator& getOperator() { return this->operator_; }
 
      const Operator& getOperator() const { return this->operator_; }

      bool refresh( const RealType& time = 0.0 )
      {
         OperatorFunction operatorFunction( this->operator_, *preimageFunction );
         this->operator_.setPreimageFunction( *this->preimageFunction );
         if( ! this->operator_.refresh( time ) ||
             ! operatorFunction.refresh( time )  )
             return false;
         this->imageFunction = operatorFunction;
         return true;
      };
 
      bool deepRefresh( const RealType& time = 0.0 )
      {
         if( ! this->preimageFunction->deepRefresh( time ) )
            return false;
         return this->refresh( time );
      };
 
      template< typename MeshEntity >
      __cuda_callable__
      RealType operator()(
         const MeshEntity& meshEntity,
         const RealType& time = 0 ) const
      {
         return imageFunction[ meshEntity.getIndex() ];
      }
 
      __cuda_callable__
      RealType operator[]( const IndexType& index ) const
      {
         return imageFunction[ index ];
      }
 
   protected:
 
      Operator& operator_;
 
      PreimageFunctionType* preimageFunction;
 
      ImageFunctionType imageFunction;
 
      template< typename, typename > friend class tnlMeshFunctionEvaluator;
};

/****
 * Specialization for precomputed evaluation and with boundary conditions.
 */
template< typename Operator,
          typename PreimageFunction,
          typename BoundaryConditions >
class tnlOperatorFunction< Operator, PreimageFunction, BoundaryConditions, false >
  : public tnlDomain< Operator::getMeshDimensions(), MeshDomain >
{
   public:
 
      static_assert( PreimageFunction::getDomainType() == MeshDomain ||
                     PreimageFunction::getDomainType() == MeshInteriorDomain ||
                     PreimageFunction::getDomainType() == MeshBoundaryDomain,
         "Only mesh preimageFunctions may be used in the operator preimageFunction. Use tnlExactOperatorFunction instead of tnlOperatorFunction." );
      static_assert( std::is_same< typename Operator::MeshType, typename PreimageFunction::MeshType >::value,
          "Both, operator and mesh preimageFunction must be defined on the same mesh." );
      static_assert( std::is_same< typename BoundaryConditions::MeshType, typename Operator::MeshType >::value,
         "The operator and the boundary conditions are defined on different mesh types." );
 
      typedef Operator OperatorType;
      typedef typename OperatorType::MeshType MeshType;
      typedef typename OperatorType::RealType RealType;
      typedef typename OperatorType::DeviceType DeviceType;
      typedef typename OperatorType::IndexType IndexType;
      typedef PreimageFunction PreimageFunctionType;
      typedef tnlMeshFunction< MeshType, Operator::getImageEntitiesDimensions() > ImageFunctionType;
      typedef BoundaryConditions BoundaryConditionsType;
      typedef tnlOperatorFunction< Operator, PreimageFunction, void, true > OperatorFunction;
      typedef typename OperatorType::ExactOperatorType ExactOperatorType;
 
      static constexpr int getEntitiesDimensions() { return OperatorType::getImageEntitiesDimensions(); };
 
      tnlOperatorFunction( OperatorType& operator_,
                           const BoundaryConditionsType& boundaryConditions,
                           const MeshType& mesh )
      :  operator_( operator_ ),
         boundaryConditions( boundaryConditions ),
         imageFunction( mesh ),
         preimageFunction( 0 )
      {};
 
      tnlOperatorFunction( OperatorType& operator_,
                           const BoundaryConditionsType& boundaryConditions,
                           const PreimageFunctionType& preimageFunction )
      :  operator_( operator_ ),
         boundaryConditions( boundaryConditions ),
         imageFunction( preimageFunction.getMesh() ),
         preimageFunction( &preimageFunction )
      {};
 
      const MeshType& getMesh() const { return imageFunction.getMesh(); };
 
      void setPreimageFunction( const PreimageFunction& preimageFunction )
      {
         this->preimageFunction = &preimageFunction;
      }
 
      const PreimageFunctionType& getPreimageFunction() const
      {
         Assert( this->preimageFunction, );
         return *this->preimageFunction;
      };
 
      PreimageFunctionType& getPreimageFunction()
      {
         Assert( this->preimageFunction, );
         return *this->preimageFunction;
      };
 
      const ImageFunctionType& getImageFunction() const { return this->imageFunction; };
 
      ImageFunctionType& getImageFunction() { return this->imageFunction; };
 
      Operator& getOperator() { return this->operator_; }
 
      const Operator& getOperator() const { return this->operator_; }

      bool refresh( const RealType& time = 0.0 )
      {
         OperatorFunction operatorFunction( this->operator_, *this->preimageFunction );
         this->operator_.setPreimageFunction( *this->preimageFunction );
         if( ! this->operator_.refresh( time ) ||
             ! operatorFunction.refresh( time )  )
             return false;
         this->imageFunction = operatorFunction;
         tnlBoundaryConditionsSetter< ImageFunctionType, BoundaryConditionsType >::apply( this->boundaryConditions, time, this->imageFunction );
         return true;
      };
 
      bool deepRefresh( const RealType& time = 0.0 )
      {
         return preimageFunction->deepRefresh( time ) &&
                this->refresh( time );
      };
 
      template< typename MeshEntity >
      __cuda_callable__
      const RealType& operator()(
         const MeshEntity& meshEntity,
         const RealType& time = 0 ) const
      {
         return imageFunction[ meshEntity.getIndex() ];
      }
 
      __cuda_callable__
      const RealType& operator[]( const IndexType& index ) const
      {
         return imageFunction[ index ];
      }
 
   protected:
 
      Operator& operator_;
 
      const PreimageFunctionType* preimageFunction;
 
      ImageFunctionType imageFunction;
 
      const BoundaryConditionsType& boundaryConditions;
 
      template< typename, typename > friend class tnlMeshFunctionEvaluator;
};

} // namespace TNL

