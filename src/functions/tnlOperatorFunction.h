/***************************************************************************
                          tnlOperatorFunction.h  -  description
                             -------------------
    begin                : Dec 31, 2015
    copyright            : (C) 2015 by oberhuber
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

#ifndef TNLOPERATORFUNCTION_H
#define	TNLOPERATORFUNCTION_H

#include <type_traits>
#include <core/tnlCuda.h>
#include <functions/tnlMeshFunction.h>

/***
 * This class evaluates given operator on given function. If the flag 
 * EvaluateOnFly is set on true, the values on particular mesh entities
 * are computed just  when operator() is called. If the EvaluateOnFly flag
 * is 'false', values on all mesh entities are evaluated by calling a method
 * refresh() they are stores in internal mesh function and the operator()
 * just returns precomputed values. If BoundaryConditions are void then the
 * values on the boundary mesh entities are undefined. In this case, the mesh
 * function evaluator evaluates this function only on the INTERIOR mesh entities.
 */

template< typename Operator,
          typename MeshFunction,
          typename BoundaryConditions = void,
          bool EvaluateOnFly = false >
class tnlOperatorFunction{};

/****
 * Specialization for 'On the fly' evaluation.
 */
template< typename Operator,
          typename MeshFunction,
          typename BoundaryConditions >
class tnlOperatorFunction< Operator, MeshFunction, BoundaryConditions, true>
 : public tnlDomain< Operator::getDimensions(), MeshDomain >
{   
   public:
      
      static_assert( MeshFunction::getDomainType() == MeshDomain ||
                     MeshFunction::getDomainType() == MeshInteriorDomain ||
                     MeshFunction::getDomainType() == MeshBoundaryDomain,
         "Only mesh functions may be used in the operator function. Use tnlExactOperatorFunction instead of tnlOperatorFunction." );
      static_assert( std::is_same< typename Operator::MeshType, typename MeshFunction::MeshType >::value,
          "Both, operator and mesh function must be defined on the same mesh." );
      static_assert( is_same< typename BoundaryConditions::MeshType, typename Operator::MeshType >::value,
         "The operator and the boundary conditions are defined on different mesh types." );
      
      typedef Operator OperatorType;
      typedef MeshFunction FunctionType;
      typedef typename OperatorType::MeshType MeshType;
      typedef typename OperatorType::RealType RealType;
      typedef typename OperatorType::DeviceType DeviceType;
      typedef typename OperatorType::IndexType IndexType;
      typedef BoundaryConditions BoundaryConditionsType;
      
      static constexpr int getEntitiesDimensions() { return OperatorType::getImageEntitiesDimensions(); };     
      
      tnlOperatorFunction(
         const OperatorType& operator_,
         const BoundaryConditionsType& boundaryConditions,
         const FunctionType& function )
      :  operator_( operator_ ), boundaryConditions( boundaryConditions ), function( function ){};
      
      void refresh() {};
      
      template< typename MeshEntity >
      __cuda_callable__
      RealType operator()(
         const MeshEntity& meshEntity,
         const RealType& time = 0 ) const
      {
         if( ! meshEntity.isBoundaryEntity() )
            return operator_( function, meshEntity, time );
         else
            return boundaryConditions( function, meshEntity, time );
      }
      
   protected:
      
      const Operator& operator_;
      
      const FunctionType& function;
      
      const BoundaryConditionsType& boundaryConditions;
      
      template< typename, typename > friend class tnlMeshFunctionEvaluator;
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
         "Only mesh functions may be used in the operator function. Use tnlExactOperatorFunction instead of tnlOperatorFunction." );
      static_assert( std::is_same< typename Operator::MeshType, typename MeshFunction::MeshType >::value,
          "Both, operator and mesh function must be defined on the same mesh." );
      
      typedef Operator OperatorType;
      typedef MeshFunction FunctionType;
      typedef typename OperatorType::MeshType MeshType;
      typedef typename OperatorType::RealType RealType;
      typedef typename OperatorType::DeviceType DeviceType;
      typedef typename OperatorType::IndexType IndexType;
      
      static constexpr int getEntitiesDimensions() { return OperatorType::getImageEntitiesDimensions(); };     
      
      tnlOperatorFunction(
         const OperatorType& operator_,
         const FunctionType& function )
      :  operator_( operator_ ), function( function ){};
      
      void refresh() {};
      
      template< typename MeshEntity >
      __cuda_callable__
      RealType operator()(
         const MeshEntity& meshEntity,
         const RealType& time = 0 ) const
      {
         return operator_( function, meshEntity, time );
      }
      
   protected:
      
      const Operator& operator_;
      
      const FunctionType& function;
      
      template< typename, typename > friend class tnlMeshFunctionEvaluator;
};

/****
 * Specialization for precomputed evaluation and no boundary conditions.
 */
template< typename Operator,
          typename MeshFunction >
class tnlOperatorFunction< Operator, MeshFunction, void, false >
 : public tnlDomain< Operator::getDimensions(), Operator::getDomainType() >
{   
   public:
      
      static_assert( MeshFunction::getDomainType() == MeshDomain ||
                     MeshFunction::getDomainType() == MeshInteriorDomain ||
                     MeshFunction::getDomainType() == MeshBoundaryDomain,
         "Only mesh functions may be used in the operator function. Use tnlExactOperatorFunction instead of tnlOperatorFunction." );
      static_assert( std::is_same< typename Operator::MeshType, typename MeshFunction::MeshType >::value,
          "Both, operator and mesh function must be defined on the same mesh." );
      
      typedef Operator OperatorType;
      typedef MeshFunction FunctionType;
      typedef typename OperatorType::MeshType MeshType;
      typedef typename OperatorType::RealType RealType;
      typedef typename OperatorType::DeviceType DeviceType;
      typedef typename OperatorType::IndexType IndexType;
      typedef tnlMeshFunction< MeshType, Operator::getImageEntitiesDimensions() > ImageFunctionType;
      typedef tnlOperatorFunction< Operator, MeshFunction, void, true > OperatorFunction;
      
      static constexpr int getEntitiesDimensions() { return OperatorType::getImageEntitiesDimensions(); };     
      
      tnlOperatorFunction(
         const OperatorType& operator_,
         const FunctionType& function )
      :  operator_( operator_ ), function( function ), imageFunction( function.getMesh() )
      {};
      
      ImageFunctionType& getImageFunction() { return this->imageFunction; };
      
      const ImageFunctionType& getImageFunction() const { return this->imageFunction; };
      
      void refresh()
      {
         this->function.refresh();
         OperatorFunction operatorFunction( this->operator_, this->function );
         this->imageFunction = operatorFunction;
      };
      
      template< typename MeshEntity >
      __cuda_callable__
      RealType operator()(
         const MeshEntity& meshEntity,
         const RealType& time = 0 ) const
      {
         return imageFunction[ meshEntity.getIndex() ];
      }
      
   protected:
      
      const Operator& operator_;
      
      const FunctionType& function;
      
      ImageFunctionType imageFunction;
      
      template< typename, typename > friend class tnlMeshFunctionEvaluator;
};

/****
 * Specialization for precomputed evaluation and with boundary conditions.
 */
template< typename Operator,
          typename MeshFunction,
          typename BoundaryConditions >
class tnlOperatorFunction< Operator, MeshFunction, BoundaryConditions, false >
  : public tnlDomain< Operator::getDimensions(), MeshDomain >
{   
   public:
      
      static_assert( MeshFunction::getDomainType() == MeshDomain ||
                     MeshFunction::getDomainType() == MeshInteriorDomain ||
                     MeshFunction::getDomainType() == MeshBoundaryDomain,
         "Only mesh functions may be used in the operator function. Use tnlExactOperatorFunction instead of tnlOperatorFunction." );
      static_assert( std::is_same< typename Operator::MeshType, typename MeshFunction::MeshType >::value,
          "Both, operator and mesh function must be defined on the same mesh." );
      static_assert( std::is_same< typename BoundaryConditions::MeshType, typename Operator::MeshType >::value,
         "The operator and the boundary conditions are defined on different mesh types." );      
      
      typedef Operator OperatorType;
      typedef MeshFunction FunctionType;
      typedef typename OperatorType::MeshType MeshType;
      typedef typename OperatorType::RealType RealType;
      typedef typename OperatorType::DeviceType DeviceType;
      typedef typename OperatorType::IndexType IndexType;
      typedef tnlMeshFunction< MeshType, Operator::getImageEntitiesDimensions() > ImageFunctionType;
      typedef BoundaryConditions BoundaryConditionsType;
      typedef tnlOperatorFunction< Operator, MeshFunction, BoundaryConditions, true > OperatorFunction;
      
      static constexpr int getEntitiesDimensions() { return OperatorType::getImageEntitiesDimensions(); };     
      
      tnlOperatorFunction(
         const OperatorType& operator_,
         const BoundaryConditionsType& boundaryConditions,
         const FunctionType& function )
      :  operator_( operator_ ), boundaryConditions( boundaryConditions ), function( function ), imageFunction( function.getMesh() )
      {};
      
      ImageFunctionType& getImageFunction() { return this->imageFunction; };
      
      const ImageFunctionType& getImageFunction() const { return this->imageFunction; };
      
      void refresh()
      {
         // TODO: Try to split it into evaluation first of interior and then of the boundary entities.
         this->function.refresh();
         OperatorFunction operatorFunction( this->operator_, this->boundaryConditions, this->function );
         this->imageFunction = operatorFunction;
      };
      
      template< typename MeshEntity >
      __cuda_callable__
      RealType operator()(
         const MeshEntity& meshEntity,
         const RealType& time = 0 ) const
      {
         return imageFunction[ meshEntity.getIndex() ];
      }
      
   protected:
      
      const Operator& operator_;
      
      const FunctionType& function;
      
      ImageFunctionType imageFunction;
      
      const BoundaryConditionsType& boundaryConditions;
      
      template< typename, typename > friend class tnlMeshFunctionEvaluator;
};



#endif	/* TNLOPERATORFUNCTION_H */

