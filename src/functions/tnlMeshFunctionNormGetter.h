/***************************************************************************
                          tnlMeshFunctionNormGetter.h  -  description
                             -------------------
    begin                : Jan 5, 2016
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

#ifndef TNLMESHFUNCTIONNORMGETTER_H
#define	TNLMESHFUNCTIONNORMGETTER_H

template< typename MeshFunction,
          typename Mesh = typename MeshFunction::MeshType >
class tnlMeshFunctionNormGetter
{
};

/***
 * Specialization for grids
 * TODO: implement this even for other devices
 */
template< int Dimensions,
          typename MeshReal,
          typename MeshIndex,
          int EntityDimensions,
          typename Real >
class tnlMeshFunctionNormGetter< tnlMeshFunction< tnlGrid< Dimensions, MeshReal, tnlHost, MeshIndex >, EntityDimensions, Real >,
                                 tnlGrid< Dimensions, MeshReal, tnlHost, MeshIndex > >
{
   public:
      
      typedef tnlMeshFunction< tnlGrid< Dimensions, MeshReal, tnlHost, MeshIndex >, EntityDimensions, Real > MeshFunctionType;
      typedef tnlGrid< Dimensions, MeshReal, tnlHost, MeshIndex > GridType;
      typedef MeshReal MeshRealType;
      typedef tnlHost DeviceType;
      typedef MeshIndex MeshIndexType;
      typedef typename MeshFunctionType::RealType RealType;
      typedef typename MeshFunctionType::MeshType MeshType;
      typedef typename MeshType::Face EntityType;
      
      static RealType getNorm( const MeshFunctionType& function,
                               const RealType& p )
      {
         if( EntityDimensions == Dimensions )
         {
            if( p == 1.0 )
               return function.getMesh().getCellMeasure() * function.getData().lpNorm( 1.0 );
            if( p == 2.0 )
               return sqrt( function.getMesh().getCellMeasure() ) * function.getData().lpNorm( 2.0 );
            return pow( function.getMesh().getCellMeasure(), 1.0 / p ) * function.getData().lpNorm( p );
         }
         if( EntityDimensions > 0 )
         {
            if( p == 1.0 )
            {
               RealType result( 0.0 );
               for( MeshIndexType i = 0;
                    i < function.getMesh().template getEntitiesCount< EntityType >();
                    i++ )
               {
                  EntityType entity = function.getMesh().template getEntity< EntityType >( i );
                  result += fabs( function[ i ] ) * entity.getMeasure();
               }
               return result;
            }
            if( p == 2.0 )
            {
               RealType result( 0.0 );
               for( MeshIndexType i = 0;
                    i < function.getMesh().template getEntitiesCount< EntityType >();
                    i++ )
               {
                  EntityType entity = function.getMesh().template getEntity< EntityType >( i );
                  result += function[ i ] * function[ i ] * entity.getMeasure();
               }            
               return sqrt( result );
            }

            RealType result( 0.0 );
            for( MeshIndexType i = 0;
                 i < function.getMesh().template getEntitiesCount< EntityType >();
                 i++ )
            {
               EntityType entity = function.getMesh().template getEntity< EntityType >( i );
               result += pow( fabs( function[ i ] ), p ) * entity.getMeasure();
            }                     
            return pow( result, 1.0 / p );
         }
         
         if( p == 1.0 )
            return function.getData().lpNorm( 1.0 );
         if( p == 2.0 )
            return function.getData().lpNorm( 2.0 );
         return function.getData().lpNorm( p );
      }
};

/****
 * Specialization for CUDA devices
 */
template< int Dimensions,
          typename MeshReal,
          typename MeshIndex,
          int EntityDimensions,
          typename Real >
class tnlMeshFunctionNormGetter< tnlMeshFunction< tnlGrid< Dimensions, MeshReal, tnlCuda, MeshIndex >, EntityDimensions, Real >,
                                 tnlGrid< Dimensions, MeshReal, tnlCuda, MeshIndex > >
{
   public:
      
      typedef tnlMeshFunction< tnlGrid< Dimensions, MeshReal, tnlCuda, MeshIndex >, EntityDimensions, Real > MeshFunctionType;
      typedef tnlGrid< Dimensions, MeshReal, tnlCuda, MeshIndex > GridType;
      typedef MeshReal MeshRealType;
      typedef tnlCuda DeviceType;
      typedef MeshIndex MeshIndexType;
      typedef typename MeshFunctionType::RealType RealType;
      typedef typename MeshFunctionType::MeshType MeshType;
      typedef typename MeshType::Face EntityType;
      
      static RealType getNorm( const MeshFunctionType& function,
                               const RealType& p )
      {
         if( EntityDimensions == Dimensions )
         {
            if( p == 1.0 )
               return function.getMesh().getCellMeasure() * function.getData().lpNorm( 1.0 );
            if( p == 2.0 )
               return sqrt( function.getMesh().getCellMeasure() ) * function.getData().lpNorm( 2.0 );
            return pow( function.getMesh().getCellMeasure(), 1.0 / p ) * function.getData().lpNorm( p );
         }
         if( EntityDimensions > 0 )
         {
            tnlAssert( false, std::cerr << "Not implemented yet." << std::endl );
         }
         
         if( p == 1.0 )
            return function.getData().lpNorm( 1.0 );
         if( p == 2.0 )
            return function.getData().lpNorm( 2.0 );
         return function.getData().lpNorm( p );
      }
};


#endif	/* TNLMESHFUNCTIONNORMGETTER_H */
