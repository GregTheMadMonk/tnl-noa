/***************************************************************************
                          DistributedGridTest.cpp  -  description
                             -------------------
    begin                : Sep 6, 2017
    copyright            : (C) 2017 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

#pragma once

#include <TNL/Functions/MeshFunction.h>

using namespace TNL;
using namespace TNL::Functions;
//=================================1D=====================================================

template <typename Real,
        int Dim>
class LinearFunction{};


template <typename Real,
        int Dim>
class ConstFunction{};

template <typename Real>
class LinearFunction<Real,1> : public Functions::Domain< 1, Functions::MeshDomain >
{
   public:
	  typedef Real RealType;
	  LinearFunction( )
	  {};

	  template< typename EntityType >
	  RealType operator()( const EntityType& meshEntity,
								  const RealType& time = 0.0 ) const
	  {
		 return meshEntity.getCenter().x();
		 
	  }
};

template <typename Real>
class ConstFunction<Real,1> : public Functions::Domain< 1, Functions::MeshDomain >
{
   public:
	  typedef Real RealType;
          
          Real Number;
          
	  ConstFunction( )
	  {};

	  template< typename EntityType >
	  RealType operator()( const EntityType& meshEntity,
								  const RealType& time = 0.0 ) const
	  {
		 return Number;
		 
	  }
};

//=================================2D======================================================

template <typename Real>
class LinearFunction<Real,2> : public Functions::Domain< 2, Functions::MeshDomain >
{
   public:
	  typedef Real RealType;
	  LinearFunction( )
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
class ConstFunction<Real,2> : public Functions::Domain< 2, Functions::MeshDomain >
{
   public:
          typedef Real RealType;
          
          Real Number;
	  ConstFunction( )
	  {};
          
	  template< typename EntityType >
	  RealType operator()( const EntityType& meshEntity,
								  const RealType& time = 0.0 ) const
	  {
		 //return meshEntity.getCoordinates().y()*10+meshEntity.getCoordinates().x();
		 return this->Number;
		 
	  }
};

//============================3D============================================================
template <typename Real>
class LinearFunction<Real,3> : public Functions::Domain< 3, Functions::MeshDomain >
{
   public:
	  typedef Real RealType;
	  LinearFunction( )
	  {};

	  template< typename EntityType >
	  RealType operator()( const EntityType& meshEntity,
								  const RealType& time = 0.0 ) const
	  {
		 //return meshEntity.getCoordinates().y()*10+meshEntity.getCoordinates().x();
		 return meshEntity.getCenter().z()*10000+meshEntity.getCenter().y()*100+meshEntity.getCenter().x();
	  }
};

template <typename Real>
class ConstFunction<Real,3> : public Functions::Domain< 3, Functions::MeshDomain >
{
   public:
          typedef Real RealType;
          
          Real Number;
	  ConstFunction( )
	  {};
          
	  template< typename EntityType >
	  RealType operator()( const EntityType& meshEntity,
								  const RealType& time = 0.0 ) const
	  {
		 //return meshEntity.getCoordinates().y()*10+meshEntity.getCoordinates().x();
		 return this->Number;
		 
	  }
};
