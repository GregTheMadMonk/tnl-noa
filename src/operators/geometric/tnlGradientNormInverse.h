/***************************************************************************
                          tnlGradientNormInverse.h  -  description
                             -------------------
    begin                : Jan 21, 2015
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

#ifndef TNLGRADIENTNORMINVERSE_H
#define	TNLGRADIENTNORMINVERSE_H

template< typename GradientNormOperator >
class tnlGradientNormInverse
   : public tnlDomain< GradientNormOperator::getDimensions(), MeshInteriorDomain >
{
   public:
      typedef GradientNormOperator GradientNormOperatorType;
      typedef typename GradientNormOperatorType::MeshType MeshType;
      typedef typename GradientNormOperatorType::RealType RealType;
            
      void setEps( const RealType& eps ) { gradientNorm.setEps( eps ); };
      
      template< typename MeshFunction, typename MeshEntity >
      __cuda_callable__
      Real operator()( const MeshFunction& u,
                       const MeshEntity& entity,
                       const Real& time = 0.0 ) const
      {
         return 1.0 / gradientNorm( u, entity, time );
      }
        
   protected:
      GradientNormOperatorType gradientNorm;
      
};

#endif	/* TNLGRADIENTNORMINVERSE_H */

