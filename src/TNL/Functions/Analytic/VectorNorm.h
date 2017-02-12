/***************************************************************************
                          VectorNorm.h  -  description
                             -------------------
    begin                : Feb 12, 2017
    copyright            : (C) 2017 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/Math.h>
#include <TNL/Assert.h>

namespace TNL {
namespace Functions {
namespace Analytic {   

template< int Dimensions_,
          typename Real >
class VectorNormBase : public Domain< Dimensions_, SpaceDomain >
{
   public:
      
      typedef Real RealType;
      typedef Containers::StaticVector< Dimensions_, RealType > VertexType;
 
      VectorNormBase()
         : center( 0.0 ),
           multiplicator( 1.0 ),
           power( 2.0 ),
           maxNorm( false ){};
           
      static void configSetup( Config::ConfigDescription& config,
                               const String& prefix = "" )
      {
         config.addEntry< double >( "center-0", "x-coordinate of the coordinates origin for the vector norm.", 0.0 );
         config.addEntry< double >( "center-1", "y-coordinate of the coordinates origin for the vector norm.", 0.0 );
         config.addEntry< double >( "center-2", "z-coordinate of the coordinates origin for the vector norm.", 0.0 );
         config.addEntry< double >( "multiplicator-0", "x-coordinate of the linear anisotropy of the vector norm.", 1.0 );
         config.addEntry< double >( "multiplicator-1", "y-coordinate of the linear anisotropy of the vector norm.", 1.0 );
         config.addEntry< double >( "multiplicator-2", "z-coordinate of the linear anisotropy of the vector norm.", 1.0 );
         config.addEntry< double >( "power", "The p coefficinet of the L-p vector norm", 2.0 );
         config.addEntry< bool >( "max-norm", "Turn to 'true' to get maximum norm.", false );
      }
 
      bool setup( const Config::ParameterContainer& parameters,
                 const String& prefix = "" )
      {
         this->power = parameters.template getParameter< double >( prefix + "power" );
         this->maxNorm = parameters.template getParameter< bool >( prefix + "infinity-power" );
         return( this->center.setup( parameters, prefix + "center-") &&
                 this->multiplicator.setup( parameters, prefix + "multiplicator-" ) );
      };

      void setCenter( const VertexType& center )
      {
         this->center = center;
      };

      const RealType& getCenter() const
      {
         return this->center;
      }
      
      void setMultiplicator( const VertexType& multiplicator )
      {
         this->multiplicator = multiplicator;
      };

      const RealType& getMultiplicator() const
      {
         return this->multiplicator;
      }
      
      void setPower( const RealType& power )
      {
         this->power = power;
      }
      
      const RealType& getPower() const
      {
         return this->power;
      }
      
      void setMaxNorm( bool maxNorm )
      {
         this->maxNorm = maxNorm;
      }
      
      const RealType& getMaxNorm() const
      {
         return this->maxNorm;
      }
      
   protected:

      VertexType center, multiplicator;
      
      RealType power;
      
      bool maxNorm;
};

template< int Dimensions,
          typename Real >
class VectorNorm
{
};

template< typename Real >
class VectorNorm< 1, Real > : public VectorNormBase< 1, Real >
{
   public:
 
      typedef VectorNormBase< 1, Real > BaseType;      
      using typename BaseType::RealType;
      using typename BaseType::VertexType;

      static String getType();

      template< int XDiffOrder = 0,
                int YDiffOrder = 0,
                int ZDiffOrder = 0 >
      __cuda_callable__
      RealType getPartialDerivative( const VertexType& v,
                                     const Real& time = 0.0 ) const
      {
         const RealType& x = v.x() - this->center.x();
         if( YDiffOrder != 0 || ZDiffOrder != 0 )
            return 0.0;
         if( XDiffOrder == 0 )
         {
            return TNL::abs( x ) * this->multiplicator.x();
         }
         if( XDiffOrder == 1 )
         {
            return TNL::sign( x ) * this->multiplicator.x();
         }
         return 0.0;
      }
      
      __cuda_callable__
      RealType operator()( const VertexType& v,
                           const RealType& time = 0.0 ) const
      {
         return this->template getPartialDerivative< 0, 0, 0 >( v, time );
      }   
};

template< typename Real >
class VectorNorm< 2, Real > : public VectorNormBase< 2, Real >
{
   public:
      
      typedef VectorNormBase< 2, Real > BaseType;      
      using typename BaseType::RealType;
      using typename BaseType::VertexType;

      static String getType();

      template< int XDiffOrder = 0,
                int YDiffOrder = 0,
                int ZDiffOrder = 0 >
      __cuda_callable__ inline
      RealType getPartialDerivative( const VertexType& v,
                                     const Real& time = 0.0 ) const
      {
         const RealType& x = v.x() - this->center.x();
         const RealType& y = v.y() - this->center.y();
         if( ZDiffOrder != 0 )
            return 0.0;
         if( XDiffOrder == 0 && YDiffOrder == 0 )
         {
            if( this->maxNorm )
               return TNL::max( TNL::abs( x ) * this->multiplicator.x(), 
                                TNL::abs( y ) * this->multiplicator.y() );
            if( this->power == 1.0 )
               return TNL::abs( x ) * this->multiplicator.x() + 
                      TNL::abs( y ) * this->multiplicator.y();
            if( this->power == 2.0 )
               return sqrt( x * x  * this->multiplicator.x() + 
                            y * y  * this->multiplicator.y() );
            return pow( pow( x, this->power ) * this->multiplicator.x() + 
                        pow( y, this->power ) * this->multiplicator.y(), 1.0 / this-> power );
         }
         TNL_ASSERT( false, "Not implemented yet." );
         return 0.0;
      }
 
      __cuda_callable__
      RealType operator()( const VertexType& v,
                           const Real& time = 0.0 ) const
      {
         return this->template getPartialDerivative< 0, 0, 0 >( v, time );
      }
};

template< typename Real >
class VectorNorm< 3, Real > : public VectorNormBase< 3, Real >
{
   public:
 
      typedef VectorNormBase< 3, Real > BaseType;      
      using typename BaseType::RealType;
      using typename BaseType::VertexType;

      static String getType();

      template< int XDiffOrder = 0,
                int YDiffOrder = 0,
                int ZDiffOrder = 0 >
      __cuda_callable__
      RealType getPartialDerivative( const VertexType& v,
                                     const Real& time = 0.0 ) const
      {
         const RealType& x = v.x() - this->center.x();
         const RealType& y = v.y() - this->center.y();
         const RealType& z = v.z() - this->center.z();
         if( XDiffOrder == 0 && YDiffOrder == 0 && ZDiffOrder == 0 )
         {
            if( this->maxNorm )
               return TNL::max( TNL::abs( x ) * this->multiplicator.x(), 
                                TNL::abs( y ) * this->multiplicator.y(),
                                TNL::abs( z ) * this->multiplicator.z() );
            if( this->power == 1.0 )
               return TNL::abs( x ) * this->multiplicator.x() + 
                      TNL::abs( y ) * this->multiplicator.y() +
                      TNL::abs( z ) * this->multiplicator.z();
            if( this->power == 2.0 )
               return sqrt( x * x  * this->multiplicator.x() + 
                            y * y  * this->multiplicator.y() +
                            z * z  * this->multiplicator.z() );
            return pow( pow( x, this->power ) * this->multiplicator.x() + 
                        pow( y, this->power ) * this->multiplicator.y() +
                        pow( z, this->power ) * this->multiplicator.z(), 1.0 / this-> power );
         }
         TNL_ASSERT( false, "Not implemented yet." );
         return 0.0;
      }
      
      __cuda_callable__
      RealType operator()( const VertexType& v,
                           const Real& time = 0.0 ) const
      {
         return this->template getPartialDerivative< 0, 0, 0 >( v, time );
      } 
};

template< int Dimensions,
          typename Real >
std::ostream& operator << ( std::ostream& str, const VectorNorm< Dimensions, Real >& f )
{
   str << "VectorNorm. function: multiplicator = " << f.getMultiplicator() << " sigma = " << f.getSigma();
   return str;
}

} // namespace Analytic
} // namespace Functions
} // namespace TNL






