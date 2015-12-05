/***************************************************************************
                          tnlFunction.h  -  description
                             -------------------
    begin                : Nov 8, 2015
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


#ifndef TNLFUNCTION_H
#define	TNLFUNCTION_H

#include <core/vectors/tnlStaticVector.h>

enum tnlFunctionType { tnlGeneralFunction, 
                       tnlDiscreteFunction,
                       tnlAnalyticFunction,
                       tnlAnalyticConstantFunction };

template< int Dimensions,
          tnlFunctionType FunctionType = tnlGeneralFunction >
class tnlFunction
{
   public:
      
      static const int dimensions = Dimensions;
      // TODO: restore constexpr when CUDA allows it
      //static constexpr int getDimensions() { return Dimensions; }
      
      //static constexpr tnlFunctionType getFunctionType() { return FunctionType; }
      enum { functionType = FunctionType };
};


/*
 * Funkce jsou bud analyticke nebo sitove( diskretni )
 *  - analyticke jsou dane vzorcem
 *    - zavisi na prostorove promenne Vertex a casu 
 *    - 
 *  - sitove jsou dane hodnotama na sitovych entitach
 *    - jejich hodnota zavisi pouze na indexu sitove entity
 * 
 * Dale jsou hybrydni funkce, ktere zavisi na vertexu, casu a indexu sitove entity.
 *   - jejich typ se urci meotdou getFunctionType
 *   - ta je def. v tnlFunction a vraci tnlHybridFunction / tnlGeneralFunction.
 *   - jde o defaultni nastaveni, ktere neni superoptimalni, ale jednoduche pro uzivatele
 * 
 */

/*
muzeme rozlisovat tnlMeshFunction a tnl(Analytic)Function
   -ta druha ma typ site void
   - chtelo by to naimplementovat tnlMeshFunction, aby se videlo, co tim lze vyresit
   - mesh-function bude mit ukazatel na mesh a dimenzi mesh entit, na nichz je definovana
   - asi nebude zaviset na case
   - mohl by pro ni byt definovany operator =, ktery by slouzil k projekci spojitych funkci na sit
   - tim bysme se zbavili enumeratoru
   - nad mesh function pak lze implementovat interpolanty - mozna vhodne i pro multigrid
      =====>>>> IMPLEMENTOVAT tnlMeshFunction
   - prozkoumat moznost lamda funkci pro mesh functions pro snazsi inicializaci sitove funkce
   nejakou pocatecni podminkou ==> mozna by to mohlo nahradit i ty analyticke funkce ???
 */
#endif	/* TNLFUNCTION_H */

