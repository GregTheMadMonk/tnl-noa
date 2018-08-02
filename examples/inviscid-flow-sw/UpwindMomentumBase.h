/***************************************************************************
                          UpwindMomentumBase.h  -  description
                             -------------------
    begin                : Feb 17, 2017
    copyright            : (C) 2017 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */


#pragma once

namespace TNL {

template< typename Mesh,
          typename Real = typename Mesh::RealType,
          typename Index = typename Mesh::IndexType >
class UpwindMomentumBase
{
   public:
      
      typedef Real RealType;
      typedef Index IndexType;
      typedef Mesh MeshType;
      typedef typename MeshType::DeviceType DeviceType;
      typedef typename MeshType::CoordinatesType CoordinatesType;
      typedef Functions::MeshFunction< MeshType > MeshFunctionType;
      static const int Dimensions = MeshType::getMeshDimension();
      typedef Functions::VectorField< Dimensions, MeshFunctionType > VelocityFieldType;
      typedef SharedPointer< MeshFunctionType > MeshFunctionPointer;
      typedef SharedPointer< VelocityFieldType > VelocityFieldPointer;
      

      void setTau(const Real& tau)
      {
          this->tau = tau;
      };

      void setGamma(const Real& gamma)
      {
          this->gamma = gamma;
      };
      
      void setVelocity( const VelocityFieldPointer& velocity )
      {
          this->velocity = velocity;
      };

      void setDensity( const MeshFunctionPointer& density )
      {
          this->density = density;
      };
      
      void setPressure( const MeshFunctionPointer& pressure )
      {
          this->pressure = pressure;
      };

      RealType positiveMainMomentumFlux( const RealType& density, const RealType& velocity, const RealType& pressure ) const
      {
         const RealType& speedOfSound = std::sqrt( this->gamma * pressure / density );
         const RealType& machNumber = velocity / speedOfSound;
         if ( machNumber <= -1.0 )
            return 0;
        else if ( machNumber <= 0.0 )
            return density * speedOfSound * speedOfSound / ( 2 * this->gamma ) * ( ( machNumber + 1.0 ) * ( machNumber + 1.0 ) );
        else if ( machNumber <= 1.0 )
            return density * speedOfSound * speedOfSound / ( 2 * this->gamma ) * ( 2.0 * ( this->gamma - 1.0 ) * machNumber  * machNumber + (machNumber + 1.0 ) * (machNumber + 1.0 ) );
        else 
            return density * velocity * velocity + pressure;
      };

      RealType negativeMainMomentumFlux( const RealType& density, const RealType& velocity, const RealType& pressure ) const
      {
         const RealType& speedOfSound = std::sqrt( this->gamma * pressure / density );
         const RealType& machNumber = velocity / speedOfSound;
         if ( machNumber <= -1.0 )
            return density * velocity * velocity + pressure;
        else if ( machNumber <= 0.0 )
            return density * speedOfSound * speedOfSound / ( 2 * this->gamma ) * ( 2.0 * ( this->gamma - 1.0 ) * machNumber * machNumber + (machNumber - 1.0 ) * (machNumber - 1.0 ) );
        else if ( machNumber <= 1.0 )
            return density * speedOfSound * speedOfSound / ( 2 * this->gamma ) * ( ( machNumber - 1.0 ) * ( machNumber - 1.0 ) );
        else 
            return 0; 
      };

      RealType positiveOtherMomentumFlux( const RealType& density, const RealType& velocity_main, const RealType& velocity_other, const RealType& pressure ) const
      {
         const RealType& speedOfSound = std::sqrt( this->gamma * pressure / density );
         const RealType& machNumber = velocity_main / speedOfSound;
         if ( machNumber <= -1.0 )
            return 0.0;
        else if ( machNumber <= 0.0 )
            return density * speedOfSound / ( 2 * this->gamma ) * ( machNumber + 1.0 ) * velocity_other;
        else if ( machNumber <= 1.0 )
            return density * speedOfSound / ( 2 * this->gamma ) * ( ( 2.0 * this->gamma - 1.0 ) * machNumber + 1.0 ) * velocity_other;
        else 
            return density * velocity_main * velocity_other;
      };

      RealType negativeOtherMomentumFlux( const RealType& density, const RealType& velocity_main, const RealType& velocity_other, const RealType& pressure ) const
      {
         const RealType& speedOfSound = std::sqrt( this->gamma * pressure / density );
         const RealType& machNumber = velocity_main / speedOfSound;
         if ( machNumber <= -1.0 )
            return density * velocity_main * velocity_other;
        else if ( machNumber <= 0.0 )
            return density * speedOfSound / ( 2 * this->gamma ) * ( ( 2.0 * this->gamma - 1.0 ) * machNumber - 1.0 ) * velocity_other;
        else if ( machNumber <= 1.0 )
            return density * speedOfSound / ( 2 * this->gamma ) * ( machNumber - 1.0 ) * velocity_other;
        else 
            return 0.0;
      };

      protected:
         
         RealType tau;

         RealType gamma;
         
         VelocityFieldPointer velocity;
         
         MeshFunctionPointer pressure;

         MeshFunctionPointer density;

};

} //namespace TNL
