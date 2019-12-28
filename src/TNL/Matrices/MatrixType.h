/***************************************************************************
                          MatrixType.h -  description
                             -------------------
    begin                : Dec 28, 2019
    copyright            : (C) 2019 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

namespace TNL {
   namespace Matrices {

struct GeneralMatrix
{
   static constexpr bool isSymmetric() { return false; }
};

struct SymmetricMatrix
{
   static constexpr bool isSymmetric() { return true; }
};

   } //namespace Matrices
} //namespace TNL