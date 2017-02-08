/***************************************************************************
                          VectorField.h  -  description
                             -------------------
    begin                : Feb 6, 2017
    copyright            : (C) 2017 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/Functions/Domain.h>
#include <TNL/Devices/Cuda.h>
#include <TNL/Config/ParameterContainer.h>
#include <TNL/Functions/MeshFunction.h>

namespace TNL {
namespace Functions {

template< int Dimensions,
          typename Function >
class VectorField 
   : public Functions::Domain< Function::getDomainDimensions(), 
                               Function::getDomainType() >
{
   public:
      
      typedef Function FunctionType;
      
      static void configSetup( Config::ConfigDescription& config,
                               const String& prefix = "" )
      {
         for( int i = 0; i < Dimensions; i++ )
            FunctionType::configSetup( config, prefix + String( i ) + "-" );
      }

      template< typename MeshPointer >
      bool setup( const MeshPointer& meshPointer,
                  const Config::ParameterContainer& parameters,
                  const String& prefix = "" )
      {
         for( int i = 0; i < Dimensions; i++ )
            if( ! vectorField[ 0 ].setup( meshPointer, parameters, prefix + String( i ) + "-" ) )
            {
               std::cerr << "Unable to setup " << i << "-th coordinate of the vector field." << std::endl;
               return false;
            }
         return true;
      }

      __cuda_callable__ 
      const FunctionType& operator[]( int i ) const
      {
         return this->vectorField[ i ];
      }
      
      __cuda_callable__ 
      FunctionType& operator[]( int i )
      {
         return this->vectorField[ i ];
      }

   protected:
      
      Containers::StaticArray< Dimensions, FunctionType > vectorField;
};
   
   
template< int Dimensions,
          typename Mesh,
          int MeshEntityDimensions,
          typename Real >
class VectorField< Dimensions, MeshFunction< Mesh, MeshEntityDimensions, Real > >
{
   public:
      
      typedef Mesh MeshType;
      typedef SharedPointer< MeshType > MeshPointer;
      typedef MeshFunction< MeshType, MeshEntityDimensions, Real > FunctionType;
      typedef typename MeshType::DeviceType DeviceType;
      typedef typename MeshType::IndexType IndexType;

      static void configSetup( Config::ConfigDescription& config,
                               const String& prefix = "" )
      {
         for( int i = 0; i < Dimensions; i++ )
            FunctionType::configSetup( config, prefix + String( i ) + "-" );
      }
      
      VectorField() {};
      
      VectorField( const MeshPointer& meshPointer )
      {
         for( int i = 0; i < Dimensions; i++ )
            this->vectorField[ i ].setMesh( meshPointer );
      };
      
      bool setup( const MeshPointer& meshPointer,
                  const Config::ParameterContainer& parameters,
                  const String& prefix = "" )
      {
         for( int i = 0; i < Dimensions; i++ )
            if( ! vectorField[ 0 ].setup( meshPointer, parameters, prefix + String( i ) + "-" ) )
            {
               std::cerr << "Unable to setup " << i << "-th coordinate of the vector field." << std::endl;
               return false;
            }
         return true;
      }

      __cuda_callable__ 
      const FunctionType& operator[]( int i ) const
      {
         return this->vectorField[ i ];
      }
      
      __cuda_callable__ 
      FunctionType& operator[]( int i )
      {
         return this->vectorField[ i ];
      }

   protected:
      
      Containers::StaticArray< Dimensions, FunctionType > vectorField;
   
};
   
} //namespace Functions
} //namepsace TNL
