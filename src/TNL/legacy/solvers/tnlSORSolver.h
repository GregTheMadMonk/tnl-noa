/***************************************************************************
                          SOROld.h  -  description
                             -------------------
    begin                : 2007/07/30
    copyright            : (C) 2007 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#ifndef SOROldH
#define SOROldH

#include <math.h>
#include <TNL/legacy/solvers/MatrixSolver.h>

template< typename Real, typename Device = Devices::Host, typename Index = int >
class SOROld : public MatrixSolver< Real, Device, Index >
{
   public:
 
   SOROld( const String& name );

   String getType() const;

   void setSOROmega( const Real& omega );

   Real getSOROmega( ) const;

   bool solve( const Matrix< Real, Device, Index >& A,
               const Vector< Real, Device, Index >& b,
               Vector< Real, Device, Index >& x,
               const Real& max_residue,
               const Index max_iterations,
               tnlPreconditioner< Real >* precond = 0 );

   protected:

   Real sorOmega;
};

template< typename Real, typename Device, typename Index >
SOROld< Real, Device, Index > :: SOROld( const String& name )
: MatrixSolver< Real, Device, Index >( name ),
  sorOmega( 1.0 )
  {
  }


template< typename Real, typename Device, typename Index >
String SOROld< Real, Device, Index > :: getType() const
{
   return String( "SOROld< " ) +
          String( getType( ( Real ) 0.0 ) ) +
          String( ", " ) +
          Device :: getDeviceType() +
          String( ", " ) +
          String( getType( ( Index ) 0 ) ) +
          String( " >" );
}

template< typename Real, typename Device, typename Index >
void SOROld< Real, Device, Index > :: setSOROmega( const Real& omega )
{
   this->sorOmega = omega;
}

template< typename Real, typename Device, typename Index >
Real SOROld< Real, Device, Index > :: getSOROmega( ) const
{
   return this->sorOmega;
}

template< typename Real, typename Device, typename Index >
bool SOROld< Real, Device, Index > :: solve( const Matrix< Real, Device, Index >& A,
                                                   const Vector< Real, Device, Index >& b,
                                                   Vector< Real, Device, Index >& x,
                                                   const Real& max_residue,
                                                   const Index max_iterations,
                                                   tnlPreconditioner< Real >* precond )
{
   const Index size = A. getSize();

   this->iteration = 0;
   this->residue = max_residue + 1.0;;

   Real bNorm = b. lpNorm( ( Real ) 2.0 );

   while( this->iteration < max_iterations &&
          max_residue < this->residue )
   {
      A. performSORIteration( this->sorOmega,
                              b,
                              x,
                              0,
                              size );
      if( this->iteration % 10 == 0 )
      {
         this->residue = this->getResidue( A, b, x, bNorm );
         if( this->verbosity > 1 )
            this->printOut();
      }
      this->iteration ++;
   }
   if( this->verbosity > 0 )
   {
      this->residue = this->getResidue( A, b, x, bNorm );
      this->printOut();
   }
   if( this->iteration <= max_iterations ) return true;
   return false;
};

#endif
