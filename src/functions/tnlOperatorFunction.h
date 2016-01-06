/***************************************************************************
                          tnlOperatorFunction.h  -  description
                             -------------------
    begin                : Dec 31, 2015
    copyright            : (C) 2015 by oberhuber
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

#ifndef TNLOPERATORFUNCTION_H
#define	TNLOPERATORFUNCTION_H

#include <type_traits>
#include <core/tnlCuda.h>

/***
 * This class evaluates given operator on given function. The image is defined only
 * for the interior mesh entities.
 */
template< typename Operator,
          typename Function >
class tnlOperatorFunction
{   
   public:
      
      typedef Operator OperatorType;
      typedef Function FunctionType;
      typedef typename OperatorType::RealType RealType;
      typedef typename OperatorType::DeviceType DeviceType;
      typedef typename OperatorType::IndexType IndexType;
      
      tnlOperatorFunction(
         const OperatorType& operator_,
         const FunctionType& function );
      
      template< typename MeshEntity >
      __cuda_callable__
      RealType operator()(
         const MeshEntity& meshEntity,
         const RealType& time );
      
   protected:
      
      const Operator* operator_;
      
      const FunctionType* function;         
};
#endif	/* TNLOPERATORFUNCTION_H */

