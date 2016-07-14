/* 
 * File:   tnlDirectEikonalMethodsBase_impl.h
 * Author: oberhuber
 *
 * Created on July 14, 2016, 3:22 PM
 */

#pragma once

#include <core/tnlTypeInfo.h>

template< typename Real,
          typename Device,
          typename Index >
   template< typename MeshFunction >
void
tnlDirectEikonalMethodsBase< tnlGrid< 1, Real, Device, Index > >::
initInterface( const MeshFunction& input,
               MeshFunction& output )
{
   const MeshType& mesh = input.getMesh();
   typedef typename MeshType::Cell Cell;
   Cell cell( mesh );
   for( cell.getCoordinates().x() = 1;
        cell.getCoordinates().x() < mesh.getDimensions().x() - 1;
        cell.getCoordinates().x() ++ )
   {
      cell.refresh();
      const auto& neighbours = cell.getNeighbourEntities();
      //const IndexType& c = cell.getIndex();
      const IndexType e = neighbours.template getEntityIndex<  1 >();
      const IndexType w = neighbours.template getEntityIndex< -1 >();
      const RealType& c = input( cell );
      if( c * input[ e ] <= 0 || c * input[ w ] <= 0 )
         output[ cell.getIndex() ] = c;
      else output[ cell.getIndex() ] =
         c > 0 ? tnlTypeInfo< RealType >::getMaxValue() :
                -tnlTypeInfo< RealType >::getMaxValue();
   }
}

template< typename Real,
          typename Device,
          typename Index >
   template< typename MeshFunction, typename MeshEntity >
void
tnlDirectEikonalMethodsBase< tnlGrid< 1, Real, Device, Index > >::
updateCell( MeshFunction& u,
            const MeshEntity& cell )
{
	const auto& neighbourEntities = cell.getNeighbourEntities< 2 >();
   const MeshType& mesh = cell.getMesh();
   
	const RealType& value = u( cell );
	Real a,b, tmp;

	if( cell.getCoordinates().x() == 0 )
		a = u[ neighbourEntities.template getEntityIndex< 1,  0 >() ];
	else if( cell.getCoordinates().x() == mesh.getDimensions().x() - 1 )
		a = u[ neighbourEntities.template getEntityIndex< -1,  0 >() ];
	else
	{
		a = fabsMin( u[ neighbourEntities.template getEntityIndex< -1,  0 >() ],
				       u[ neighbourEntities.template getEntityIndex<  1,  0 >() ] );
	}

	if( cell.getCoordinates().y() == 0 )
		b = u[ neighbourEntities.template getEntityIndex< 0,  1 >()];
	else if( cell.getCoordinates().y() == mesh.getDimensions().y() - 1 )
		b = u[ neighbourEntities.template getEntityIndex< 0,  -1 >() ];
	else
	{
		b = fabsMin( u[ neighbourEntities.template getEntityIndex< 0,  -1 >() ],
				       u[ neighbourEntities.template getEntityIndex< 0,   1 >() ] );
	}

	if( fabs( a - b ) >= h )
		tmp = fabsMin( a, b ) + Sign( value ) * h;
	else
		tmp = 0.5 * (a + b + Sign(value)*sqrt(2.0 * h * h - (a - b) * (a - b) ) );
	
   u[ cell.getIndex() ] = fabsMin( value, tmp );
}


template< typename Real,
          typename Device,
          typename Index >
      template< typename MeshFunction >
void
tnlDirectEikonalMethodsBase< tnlGrid< 2, Real, Device, Index > >::
initInterface( const MeshFunction& input,
               MeshFunction& output )
{
   const MeshType& mesh = input.getMesh();
   typedef typename MeshType::Cell Cell;
   Cell cell( mesh );
   for( cell.getCoordinates().y() = 0;
        cell.getCoordinates().y() < mesh.getDimensions().y();
        cell.getCoordinates().y() ++ )
      for( cell.getCoordinates().x() = 0;
           cell.getCoordinates().x() < mesh.getDimensions().x();
           cell.getCoordinates().x() ++ )
      {
         cell.refresh();
         const RealType& c = input( cell );
         if( ! cell.isBoundaryEntity()  )
         {
            auto neighbours = cell.getNeighbourEntities();
            const IndexType e = neighbours.template getEntityIndex<  1,  0 >();
            const IndexType w = neighbours.template getEntityIndex< -1,  0 >();
            const IndexType n = neighbours.template getEntityIndex<  0,  1 >();
            const IndexType s = neighbours.template getEntityIndex<  0, -1 >();            
            if( c * input[ e ] <= 0 || c * input[ w ] <= 0 ||
                c * input[ n ] <= 0 || c * input[ s ] <= 0 )
            {
               output[ cell.getIndex() ] = c;
               continue;
            }
         }
         output[ cell.getIndex() ] =
            c > 0 ? tnlTypeInfo< RealType >::getMaxValue() :
                   -tnlTypeInfo< RealType >::getMaxValue();         
      }
}

template< typename Real,
          typename Device,
          typename Index >
   template< typename MeshFunction, typename MeshEntity >
void
tnlDirectEikonalMethodsBase< tnlGrid< 2, Real, Device, Index > >::
updateCell( MeshFunction& u,
            const MeshEntity& cell )
{
   
}


template< typename Real,
          typename Device,
          typename Index >
   template< typename MeshFunction >
void
tnlDirectEikonalMethodsBase< tnlGrid< 3, Real, Device, Index > >::
initInterface( const MeshFunction& input,
               MeshFunction& output )
{
   const MeshType& mesh = input.getMesh();
   typedef typename MeshType::Cell Cell;
   Cell cell( mesh );
   for( cell.getCoordinates().z() = 1;
        cell.getCoordinates().z() < mesh.getDimensions().z() - 1;
        cell.getCoordinates().z() ++ )   
      for( cell.getCoordinates().y() = 1;
           cell.getCoordinates().y() < mesh.getDimensions().y() - 1;
           cell.getCoordinates().y() ++ )
         for( cell.getCoordinates().x() = 1;
              cell.getCoordinates().x() < mesh.getDimensions().x() - 1;
              cell.getCoordinates().x() ++ )
         {
            cell.refresh();
            auto neighbours = cell.getNeighbourEntities();
            //const IndexType& c = cell.getIndex();
            const IndexType e = neighbours.template getEntityIndex<  1,  0,  0 >();
            const IndexType w = neighbours.template getEntityIndex< -1,  0,  0 >();
            const IndexType n = neighbours.template getEntityIndex<  0,  1,  0 >();
            const IndexType s = neighbours.template getEntityIndex<  0, -1,  0 >();
            const IndexType t = neighbours.template getEntityIndex<  0,  0,  1 >();
            const IndexType b = neighbours.template getEntityIndex<  0,  0, -1 >();
            const RealType& c = input( cell );
            if( c * input[ e ] <= 0 || c * input[ w ] <= 0 ||
                c * input[ n ] <= 0 || c * input[ s ] <= 0 ||
                c * input[ t ] <= 0 || c * input[ b ] <= 0 )
               output[ cell.getIndex() ] = c;
            else output[ cell.getIndex() ] =
               c > 0 ? tnlTypeInfo< RealType >::getMaxValue() :
                      -tnlTypeInfo< RealType >::getMaxValue();
         }
}

template< typename Real,
          typename Device,
          typename Index >
   template< typename MeshFunction, typename MeshEntity >
void
tnlDirectEikonalMethodsBase< tnlGrid< 3, Real, Device, Index > >::
updateCell( MeshFunction& u,
            const MeshEntity& cell )
{
   
}
