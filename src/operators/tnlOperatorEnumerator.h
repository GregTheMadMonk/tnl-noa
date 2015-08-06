/***************************************************************************
                          tnlOperatorEnumerator.h  -  description
                             -------------------
    begin                : Mar 8, 2015
    copyright            : (C) 2015 by Tomas Oberhuber
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
#ifndef SRC_OPERATORS_TNLOPERATORENUMERATOR_H_
#define SRC_OPERATORS_TNLOPERATORENUMERATOR_H_

//#include <_operators/tnlOperatorAdapter.h>

template< typename Operator,
          typename DofVector >
class tnlOperatorEnumeratorTraverserUserData
{
   public:

      typedef typename DofVector::RealType RealType;

      const RealType *time;

      const Operator* _operator;

      DofVector *u;

      const RealType* _operatorCoefficient;

      const RealType* dofVectorCoefficient;

      tnlOperatorEnumeratorTraverserUserData( const RealType& time,
                                              const Operator& _operator,
                                              DofVector& u,
                                              const RealType& _operatorCoefficient,
                                              const RealType& dofVectorCoefficient )
      : time( &time ),
        _operator( &_operator ),
        u( &u ),
        _operatorCoefficient( &_operatorCoefficient ),
        dofVectorCoefficient( &dofVectorCoefficient )
      {};
};


template< typename Mesh,
          typename Operator,
          typename DofVector >
class tnlOperatorEnumerator
{
   public:
      typedef Mesh MeshType;
      typedef typename DofVector::RealType RealType;
      typedef typename DofVector::DeviceType DeviceType;
      typedef typename DofVector::IndexType IndexType;
      typedef tnlOperatorEnumeratorTraverserUserData< Operator,
                                                      DofVector > TraverserUserData;

      template< int EntityDimensions >
      void enumerate( const MeshType& mesh,
                      const Operator& _operator,
                      DofVector& u,
                      const RealType& _operatorCoefficient = 1.0,
                      const RealType& dofVectorCoefficient = 0.0,
                      const RealType& time = 0.0 ) const;


      class TraverserEntitiesProcessor
      {
         public:

            template< int EntityDimensions >
#ifdef HAVE_CUDA
            __host__ __device__
#endif
            static void processEntity( const MeshType& mesh,
                                       TraverserUserData& userData,
                                       const IndexType index )
            {
               //typedef tnlOperatorAdapter< MeshType, Operator > OperatorAdapter;
               ( *userData.u )[ index ] =
                        ( *userData.dofVectorCoefficient ) * ( *userData.u )[ index ] +
                        ( *userData._operatorCoefficient ) * userData._operator ->getValue( mesh,
                                                                                            index,
                                                                                            *userData.time );
            }

      };

};

template< int Dimensions,
          typename Real,
          typename Device,
          typename Index,
          typename Operator,
          typename DofVector >
class tnlOperatorEnumerator< tnlGrid< Dimensions, Real, Device, Index >,
                             Operator,
                             DofVector >
{
   public:

      typedef tnlGrid< Dimensions, Real, Device, Index > MeshType;
      typedef typename MeshType::RealType RealType;
      typedef typename MeshType::DeviceType DeviceType;
      typedef typename MeshType::IndexType IndexType;
      typedef typename MeshType::CoordinatesType CoordinatesType;
      typedef tnlOperatorEnumeratorTraverserUserData< Operator,
                                                      DofVector > TraverserUserData;

      template< int EntityDimensions >
      void enumerate( const MeshType& mesh,
                      const Operator& _operator,
                      DofVector& u,
                      const RealType& _operatorCoefficient = 1.0,
                      const RealType& dofVectorCoefficient = 0.0,
                      const RealType& time = 0.0 ) const;

      class TraverserEntitiesProcessor
      {
         public:

         typedef typename MeshType::VertexType VertexType;

#ifdef HAVE_CUDA
            __host__ __device__
#endif
            static void processCell( const MeshType& mesh,
                                     TraverserUserData& userData,
                                     const IndexType index,
                                     const CoordinatesType& coordinates )
            {
               //printf( "Enumerator::processCell mesh =%p \n", &mesh );
               //typedef tnlOperatorAdapter< MeshType, Operator > OperatorAdapter;
               ( *userData.u )[ index ] =
                        ( *userData.dofVectorCoefficient ) * ( *userData.u )[ index ] +
                        ( *userData._operatorCoefficient ) * userData._operator->getValue( mesh,
                                                                                           index,
                                                                                           coordinates,
                                                                                           *userData.time );

            }

#ifdef HAVE_CUDA
            __host__ __device__
#endif
            static void processFace( const MeshType& mesh,
                                     TraverserUserData& userData,
                                     const IndexType index,
                                     const CoordinatesType& coordinates )
            {
               //typedef tnlOperatorAdapter< MeshType, Operator > OperatorAdapter;
               ( *userData.u )[ index ] =
                        ( *userData.dofVectorCoefficient ) * ( *userData.u )[ index ] +
                        ( *userData._operatorCoefficient ) * userData._operator->getValue( mesh,
                                                                                           index,
                                                                                           coordinates,
                                                                                           *userData.time );
            }
      };

};

#include <operators/tnlOperatorEnumerator_impl.h>

#endif /* SRC_OPERATORS_TNLOPERATORENUMERATOR_H_ */
