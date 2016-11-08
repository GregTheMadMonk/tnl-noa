/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   tnlFastSweepingMethod2D_impl.h
 * Author: oberhuber
 *
 * Created on July 14, 2016, 10:32 AM
 */

#pragma once

#include "tnlFastSweepingMethod.h"

template< typename Real,
          typename Device,
          typename Index,
          typename Anisotropy >
tnlFastSweepingMethod< tnlGrid< 2, Real, Device, Index >, Anisotropy >::
tnlFastSweepingMethod()
: maxIterations( 1 )
{
   
}

template< typename Real,
          typename Device,
          typename Index,
          typename Anisotropy >
const Index&
tnlFastSweepingMethod< tnlGrid< 2, Real, Device, Index >, Anisotropy >::
getMaxIterations() const
{
   
}

template< typename Real,
          typename Device,
          typename Index,
          typename Anisotropy >
void
tnlFastSweepingMethod< tnlGrid< 2, Real, Device, Index >, Anisotropy >::
setMaxIterations( const IndexType& maxIterations )
{
   
}

template< typename Real,
          typename Device,
          typename Index,
          typename Anisotropy >
void
tnlFastSweepingMethod< tnlGrid< 2, Real, Device, Index >, Anisotropy >::
solve( const MeshType& mesh,
       const AnisotropyType& anisotropy,
       MeshFunctionType& u )
{
   MeshFunctionType aux;
   InterfaceMapType interfaceMap;
   aux.setMesh( mesh );
   interfaceMap.setMesh( mesh );
   std::cout << "Initiating the interface cells ..." << std::endl;
   BaseType::initInterface( u, aux, interfaceMap );
   //aux.save( "aux-ini.tnl" );

   typename MeshType::Cell cell( mesh );
   
   IndexType iteration( 0 );
   while( iteration < this->maxIterations )
   {

      for( cell.getCoordinates().y() = 0;
           cell.getCoordinates().y() < mesh.getDimensions().y();
           cell.getCoordinates().y()++ )
      {
         for( cell.getCoordinates().x() = 0;
              cell.getCoordinates().x() < mesh.getDimensions().x();
              cell.getCoordinates().x()++ )
            {
               cell.refresh();
               if( ! interfaceMap( cell ) )
                  this->updateCell( aux, cell );
            }
      }
      //aux.save( "aux-1.tnl" );

      for( cell.getCoordinates().y() = 0;
           cell.getCoordinates().y() < mesh.getDimensions().y();
           cell.getCoordinates().y()++ )
      {
         for( cell.getCoordinates().x() = mesh.getDimensions().x() - 1;
              cell.getCoordinates().x() >= 0 ;
              cell.getCoordinates().x()-- )		
            {
               //std::cerr << "2 -> ";
               cell.refresh();
               if( ! interfaceMap( cell ) )            
                  this->updateCell( aux, cell );
            }
      }
      //aux.save( "aux-2.tnl" );

      for( cell.getCoordinates().y() = mesh.getDimensions().y() - 1;
           cell.getCoordinates().y() >= 0 ;
           cell.getCoordinates().y()-- )
         {
         for( cell.getCoordinates().x() = 0;
              cell.getCoordinates().x() < mesh.getDimensions().x();
              cell.getCoordinates().x()++ )
            {
               //std::cerr << "3 -> ";
               cell.refresh();
               if( ! interfaceMap( cell ) )            
                  this->updateCell( aux, cell );
            }
         }
      //aux.save( "aux-3.tnl" );


      for( cell.getCoordinates().y() = mesh.getDimensions().y() - 1;
           cell.getCoordinates().y() >= 0;
           cell.getCoordinates().y()-- )
         {
         for( cell.getCoordinates().x() = mesh.getDimensions().x() - 1;
              cell.getCoordinates().x() >= 0 ;
              cell.getCoordinates().x()-- )		
            {
               //std::cerr << "4 -> ";
               cell.refresh();
               if( ! interfaceMap( cell ) )            
                  this->updateCell( aux, cell );
            }
         }   
      //aux.save( "aux-4.tnl" );
      iteration++;
   }
}

