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
      typedef Containers::StaticVector< Dimensions_, RealType > PointType;
 
      VectorNormBase()
         : center( 0.0 ),
           anisotropy( 1.0 ),
           power( 2.0 ),
           radius( 0.0 ),
           multiplicator( 1.0 ),
           maxNorm( false ){};
           
      static void configSetup( Config::ConfigDescription& config,
                               const String& prefix = "" )
      {
         config.addEntry< double >( prefix + "center-0", "x-coordinate of the coordinates origin for the vector norm.", 0.0 );
         config.addEntry< double >( prefix + "center-1", "y-coordinate of the coordinates origin for the vector norm.", 0.0 );
         config.addEntry< double >( prefix + "center-2", "z-coordinate of the coordinates origin for the vector norm.", 0.0 );
         config.addEntry< double >( prefix + "anisotropy-0", "x-coordinate of the linear anisotropy of the vector norm.", 1.0 );
         config.addEntry< double >( prefix + "anisotropy-1", "y-coordinate of the linear anisotropy of the vector norm.", 1.0 );
         config.addEntry< double >( prefix + "anisotropy-2", "z-coordinate of the linear anisotropy of the vector norm.", 1.0 );
         config.addEntry< double >( prefix + "power", "The p coefficient of the L-p vector norm", 2.0 );
         config.addEntry< double >( prefix + "radius", "Radius of the zero-th level-set.", 0.0 );
         config.addEntry< double >( prefix + "multiplicator", "Outer multiplicator of the norm - -1.0 turns the function graph upside/down.", 1.0 );
         config.addEntry< bool >( prefix + "max-norm", "Turn to 'true' to get maximum norm.", false );
      }
 
      bool setup( const Config::ParameterContainer& parameters,
                  const String& prefix = "" )
      {
         this->power = parameters.template getParameter< double >( prefix + "power" );
         this->maxNorm = parameters.template getParameter< bool >( prefix + "max-norm" );
         this->radius = parameters.template getParameter< double >( prefix + "radius" );
         this->multiplicator = parameters.template getParameter< double >( prefix + "multiplicator" );
         return( this->center.setup( parameters, prefix + "center-") &&
                 this->anisotropy.setup( parameters, prefix + "anisotropy-" ) );
      };

      void setCenter( const PointType& center )
      {
         this->center = center;
      };

      const RealType& getCenter() const
      {
         return this->center;
      }
      
      void setAnisotropy( const PointType& anisotropy )
      {
         this->anisotropy = anisotropy;
      };

      const RealType& getAnisotropy() const
      {
         return this->anisotropy;
      }
      
      void setPower( const RealType& power )
      {
         this->power = power;
      }
      
      const RealType& getPower() const
      {
         return this->power;
      }
      
      void setRadius( const RealType& radius )
      {
         this->radius = radius;
      }
      
      const RealType& getRadius() const
      {
         return this->radius;
      }
      
      void setMultiplicator( const RealType& multiplicator )
      {
         this->multiplicator = multiplicator;
      }
      
      const RealType& getMultiplicator() const
      {
         return this->multiplicator;
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

      PointType center, anisotropy;
      
      RealType power, radius, multiplicator;
      
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
      using typename BaseType::PointType;

      static String getType();

      template< int XDiffOrder = 0,
                int YDiffOrder = 0,
                int ZDiffOrder = 0 >
      __cuda_callable__
      RealType getPartialDerivative( const PointType& v,
                                     const Real& time = 0.0 ) const
      {
         const RealType& x = v.x() - this->center.x();
         if( YDiffOrder != 0 || ZDiffOrder != 0 )
            return 0.0;
         if( XDiffOrder == 0 )
         {
            return this->multiplicator * ( TNL::abs( x ) * this->anisotropy.x() - this->radius );
         }
         if( XDiffOrder == 1 )
         {
            return this->multiplicator * TNL::sign( x ) * this->anisotropy.x();
         }
         return 0.0;
      }
      
      __cuda_callable__
      RealType operator()( const PointType& v,
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
      using typename BaseType::PointType;

      static String getType();

      template< int XDiffOrder = 0,
                int YDiffOrder = 0,
                int ZDiffOrder = 0 >
      __cuda_callable__ inline
      RealType getPartialDerivative( const PointType& v,
                                     const Real& time = 0.0 ) const
      {
         const RealType& x = v.x() - this->center.x();
         const RealType& y = v.y() - this->center.y();
         if( ZDiffOrder != 0 )
            return 0.0;
         if( XDiffOrder == 0 && YDiffOrder == 0 )
         {
            if( this->maxNorm )
               return ( TNL::max( TNL::abs( x ) * this->anisotropy.x(), 
                                  TNL::abs( y ) * this->anisotropy.y() ) - this->radius ) * this->multiplicator;
            if( this->power == 1.0 )
               return ( ( TNL::abs( x ) * this->anisotropy.x() + 
                          TNL::abs( y ) * this->anisotropy.y() ) - this->radius ) * this->multiplicator;
            if( this->power == 2.0 )
               return ( std::sqrt( x * x  * this->anisotropy.x() + 
                                   y * y  * this->anisotropy.y() ) - this->radius ) * this->multiplicator;
            return ( std::pow( std::pow( TNL::abs( x ), this->power ) * this->anisotropy.x() + 
                               std::pow( TNL::abs( y ), this->power ) * this->anisotropy.y(), 1.0 / this-> power ) - this->radius ) * this->multiplicator;
         }
         TNL_ASSERT_TRUE( false, "Not implemented yet." );
         return 0.0;
      }
 
      __cuda_callable__
      RealType operator()( const PointType& v,
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
      using typename BaseType::PointType;

      static String getType();

      template< int XDiffOrder = 0,
                int YDiffOrder = 0,
                int ZDiffOrder = 0 >
      __cuda_callable__
      RealType getPartialDerivative( const PointType& v,
                                     const Real& time = 0.0 ) const
      {
         const RealType& x = v.x() - this->center.x();
         const RealType& y = v.y() - this->center.y();
         const RealType& z = v.z() - this->center.z();
         if( XDiffOrder == 0 && YDiffOrder == 0 && ZDiffOrder == 0 )
         {
            if( this->maxNorm )
               return ( TNL::max( TNL::max( TNL::abs( x ) * this->anisotropy.x(), 
                                            TNL::abs( y ) * this->anisotropy.y() ),
                                  TNL::abs( z ) * this->anisotropy.z() ) - this->radius ) * this->multiplicator;
            if( this->power == 1.0 )
               return ( ( TNL::abs( x ) * this->anisotropy.x() + 
                          TNL::abs( y ) * this->anisotropy.y() +
                          TNL::abs( z ) * this->anisotropy.z() ) - this->radius ) * this->multiplicator;
            if( this->power == 2.0 )
               return ( std::sqrt( x * x  * this->anisotropy.x() + 
                                   y * y  * this->anisotropy.y() +
                                   z * z  * this->anisotropy.z() ) - this->radius ) * this->multiplicator ;
            return ( std::pow( std::pow( TNL::abs( x ), this->power ) * this->anisotropy.x() + 
                               std::pow( TNL::abs( y ), this->power ) * this->anisotropy.y() +
                               std::pow( TNL::abs( z ), this->power ) * this->anisotropy.z(), 1.0 / this-> power ) - this->radius ) * this->multiplicator;
         }
         TNL_ASSERT_TRUE( false, "Not implemented yet." );
         return 0.0;
      }
      
      __cuda_callable__
      RealType operator()( const PointType& v,
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






