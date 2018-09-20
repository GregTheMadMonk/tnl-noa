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
#include <TNL/Functions/VectorFieldGnuplotWriter.h>
#include <TNL/Functions/VectorFieldVTKWriter.h>

namespace TNL {
namespace Functions {

template< int Size,
          typename Function >
class VectorField 
   : public Functions::Domain< Function::getDomainDimension(), 
                               Function::getDomainType() >
{
   public:
      
      typedef Function FunctionType;
      typedef typename FunctionType::RealType RealType;
      typedef typename FunctionType::PointType PointType;
      
      static void configSetup( Config::ConfigDescription& config,
                               const String& prefix = "" )
      {
         for( int i = 0; i < Size; i++ )
            FunctionType::configSetup( config, prefix + String( i ) + "-" );
      }

      template< typename MeshPointer >
      bool setup( const MeshPointer& meshPointer,
                  const Config::ParameterContainer& parameters,
                  const String& prefix = "" )
      {
         for( int i = 0; i < Size; i++ )
            if( ! vectorField[ i ].setup( parameters, prefix + String( i ) + "-" ) )
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
      
      Containers::StaticArray< Size, FunctionType > vectorField;
};
   
   
template< int Size,
          typename Mesh,
          int MeshEntityDimension,
          typename Real >
class VectorField< Size, MeshFunction< Mesh, MeshEntityDimension, Real > >
: public Functions::Domain< MeshFunction< Mesh, MeshEntityDimension, Real >::getDomainDimension(), 
                            MeshFunction< Mesh, MeshEntityDimension, Real >::getDomainType() >,
   public Object
{
   public:
      
      typedef Mesh MeshType;
      typedef Real RealType;
      typedef Pointers::SharedPointer<  MeshType > MeshPointer;
      typedef MeshFunction< MeshType, MeshEntityDimension, RealType > FunctionType;
      typedef Pointers::SharedPointer<  FunctionType > FunctionPointer;
      typedef typename MeshType::DeviceType DeviceType;
      typedef typename MeshType::GlobalIndexType IndexType;
      typedef VectorField< Size, MeshFunction< Mesh, MeshEntityDimension, RealType > > ThisType;
      typedef Containers::StaticVector< Size, RealType > VectorType;
      
      static constexpr int getEntitiesDimension() { return FunctionType::getEntitiesDimension(); }
      
      static constexpr int getMeshDimension() { return MeshType::getMeshDimension(); }

	  static constexpr int getVectorDimension() { return Size; }
      

      static void configSetup( Config::ConfigDescription& config,
                               const String& prefix = "" )
      {
         for( int i = 0; i < Size; i++ )
            FunctionType::configSetup( config, prefix + String( i ) + "-" );
      }
      
      VectorField() {};
      
      VectorField( const MeshPointer& meshPointer )
      {
         for( int i = 0; i < Size; i++ )
            this->vectorField[ i ]->setMesh( meshPointer );
      };
      
      static String getType()
      {
         return String( "Functions::VectorField< " ) +
                  String( Size) + ", " +
                 FunctionType::getType() +
                  " >";
      }
 
      String getTypeVirtual() const
      {
         return this->getType();
      }
 
      static String getSerializationType()
      {
         return String( "Functions::VectorField< " ) +
                  String( Size) + ", " +
                 FunctionType::getSerializationType() +
                  " >";         
      }

      virtual String getSerializationTypeVirtual() const
      {
         return this->getSerializationType();
      }
      
      
      void setMesh( const MeshPointer& meshPointer )
      {
         for( int i = 0; i < Size; i++ )
            this->vectorField[ i ]->setMesh( meshPointer );
      }
      
      template< typename Device = Devices::Host >
      __cuda_callable__
      const MeshType& getMesh() const
      {
         return this->vectorField[ 0 ].template getData< Device >().template getMesh< Device >();
      }

      const MeshPointer& getMeshPointer() const
      {
         return this->vectorField[ 0 ]->getMeshPointer();
      }
      
      bool setup( const MeshPointer& meshPointer,
                  const Config::ParameterContainer& parameters,
                  const String& prefix = "" )
      {
         for( int i = 0; i < Size; i++ )
            if( ! vectorField[ i ].setup( meshPointer, parameters, prefix + String( i ) + "-" ) )
            {
               std::cerr << "Unable to setup " << i << "-th coordinate of the vector field." << std::endl;
               return false;
            }
         return true;
      }
      
      static IndexType getDofs( const MeshPointer& meshPointer )
      {
         return Size * FunctionType::getDofs( meshPointer );
      }
      
      void bind( ThisType& vectorField )
      {
         for( int i = 0; i < Size; i ++ )
         {
            this->vectorField[ i ]->bind( vectorField[ i ] );
         }
      };
 
      template< typename Vector >
      void bind( const MeshPointer& meshPointer,
                 const Vector& data,
                 IndexType offset = 0 )
      {
         TNL_ASSERT_GE( data.getSize(), offset + Size * this->vectorField[ 0 ]->getDofs( meshPointer ),
                        "Attempt to bind vector which is not large enough."  );
         for( int i = 0; i < Size; i ++ )
         {
            this->vectorField[ i ]->bind( meshPointer, data, offset );
            offset += this->vectorField[ i ]->getDofs(meshPointer);
         }
      }
      
      template< typename Vector >
      void bind( const MeshPointer& meshPointer,
                 const Pointers::SharedPointer< Vector >& dataPtr,
                 IndexType offset = 0 )
      {
         TNL_ASSERT_GE( dataPtr->getSize(), offset + Size * this->vectorField[ 0 ]->getDofs( meshPointer ),
                        "Attempt to bind vector which is not large enough." );
         for( int i = 0; i < Size; i ++ )
         {
            this->vectorField[ i ]->bind( meshPointer, dataPtr, offset );
            offset += this->vectorField[ i ]->getDofs( meshPointer );
         }         
      }

      __cuda_callable__ 
      const FunctionPointer& operator[]( int i ) const
      {
         return this->vectorField[ i ];
      }
      
      __cuda_callable__ 
      FunctionPointer& operator[]( int i )
      {
         return this->vectorField[ i ];
      }
      
      __cuda_callable__ 
      void setElement( const IndexType i, const VectorType& v )
      {
         for( int j = 0; j < Size; j++ )
            ( *this )[ j ]->getData().setElement( i, v[ j ] );
      }
      
      __cuda_callable__
      VectorType getElement( const IndexType i ) const
      {
         VectorType v;
         for( int j = 0; j < Size; j++ )
            v[ j ] = ( *this )[ j ]->getData().getElement( i );
         return v;
      }
      
      __cuda_callable__
      VectorType getVector( const IndexType index ) const
      {
         VectorType v;
         for( int i = 0; i < Size; i++ )
            // FIXME: fix the dereferencing operator in smart pointers to be __cuda_callable__
            v[ i ] = this->vectorField[ i ].template getData< Devices::Cuda >()[ index ];
         return v;
      }

      template< typename EntityType >
      void setValue( const EntityType& meshEntity,
                     const FunctionType& value )
      {
         for(int i = 0; i < Size; i++ )
            this->vectorfield[ i ].setValue( meshEntity.getIndex(), value[ i ] );
      }

      template< typename EntityType >
      __cuda_callable__
      VectorType getVector( const EntityType& meshEntity ) const
      {
         VectorType v;
         for( int i = 0; i < Size; i++ )
            // FIXME: fix the dereferencing operator in smart pointers to be __cuda_callable__
            v[ i ] = this->vectorField[ i ].template getData< Devices::Cuda >()( meshEntity );
         return v;
      }
      
      bool save( File& file ) const
      {
         if( ! Object::save( file ) )
            return false;
         for( int i = 0; i < Size; i++ )
            if( ! vectorField[ i ]->save( file ) )
               return false;
         return true;
      }

      bool load( File& file )
      {
         if( ! Object::load( file ) )
            return false;
         for( int i = 0; i < Size; i++ )
            if( ! vectorField[ i ]->load( file ) )
               return false;
         return true;
      }
 
      bool boundLoad( File& file )
      {
         if( ! Object::load( file ) )
            return false;
         for( int i = 0; i < Size; i++ )
            if( ! vectorField[ i ]->boundLoad( file ) )
               return false;
         return true;         
      }
      
      bool write( const String& fileName,
                  const String& format = "vtk",
                  const double& scale = 1.0 ) const
      {
         std::fstream file;
         file.open( fileName.getString(), std::ios::out );
         if( ! file )
         {
            std::cerr << "Unable to open a file " << fileName << "." << std::endl;
            return false;
         }
         if( format == "vtk" )
            return VectorFieldVTKWriter< ThisType >::write( *this, file, scale );
         else if( format == "gnuplot" )
            return VectorFieldGnuplotWriter< ThisType >::write( *this, file, scale );
         else {
            std::cerr << "Unknown output format: " << format << std::endl;
            return false;
         }
         return true;
      }
      
      using Object::save;
 
      using Object::load;
 
      using Object::boundLoad;      

   protected:
      
      Containers::StaticArray< Size, FunctionPointer > vectorField;
   
};

template< int Dimension,
          typename Function >
std::ostream& operator << ( std::ostream& str, const VectorField< Dimension, Function >& f )
{
   for( int i = 0; i < Dimension; i++ )
   {
      str << "[ " << f[ i ] << " ]";
      if( i < Dimension - 1 )
         str << ", ";
   }
   return str;
}

   
} //namespace Functions
} //namepsace TNL
