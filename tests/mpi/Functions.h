/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   Functions.h
 * Author: slimon
 *
 * Created on 29. března 2017, 17:10
 */

#pragma once

#include <TNL/Functions/MeshFunction.h>

using namespace TNL;
using namespace TNL::Functions;
//=================================1D=====================================================

template <typename Real,
        int Dim>
class FunctionToEvaluate{};


template <typename Real,
        int Dim>
class ZeroFunction{};

template <typename Real>
class FunctionToEvaluate<Real,1> : public Functions::Domain< 1, Functions::MeshDomain >
{
   public:
	  typedef Real RealType;
	  FunctionToEvaluate( )
	  {};

	  template< typename EntityType >
	  RealType operator()( const EntityType& meshEntity,
								  const RealType& time = 0.0 ) const
	  {
		 //return meshEntity.getCoordinates().y()*10+meshEntity.getCoordinates().x();
		 return meshEntity.getCenter().x();
		 
	  }
};

template <typename Real>
class ZeroFunction<Real,1> : public Functions::Domain< 1, Functions::MeshDomain >
{
   public:
	  typedef Real RealType;
	  ZeroFunction( )
	  {};

	  template< typename EntityType >
	  RealType operator()( const EntityType& meshEntity,
								  const RealType& time = 0.0 ) const
	  {
		 //return meshEntity.getCoordinates().y()*10+meshEntity.getCoordinates().x();
		 return -1.0;
		 
	  }
};

//=================================2D======================================================

template <typename Real>
class FunctionToEvaluate<Real,2> : public Functions::Domain< 2, Functions::MeshDomain >
{
   public:
	  typedef Real RealType;
	  FunctionToEvaluate( )
	  {};

	  template< typename EntityType >
	  RealType operator()( const EntityType& meshEntity,
								  const RealType& time = 0.0 ) const
	  {
		 //return meshEntity.getCoordinates().y()*10+meshEntity.getCoordinates().x();
		 return meshEntity.getCenter().y()*100+meshEntity.getCenter().x();
              
		 
	  }
};

template <typename Real>
class ZeroFunction<Real,2> : public Functions::Domain< 2, Functions::MeshDomain >
{
   public:
          typedef Real RealType;
          
          Real Number;
	  ZeroFunction( )
	  {};
          
	  template< typename EntityType >
	  RealType operator()( const EntityType& meshEntity,
								  const RealType& time = 0.0 ) const
	  {
		 //return meshEntity.getCoordinates().y()*10+meshEntity.getCoordinates().x();
		 return this->Number;
		 
	  }
};
