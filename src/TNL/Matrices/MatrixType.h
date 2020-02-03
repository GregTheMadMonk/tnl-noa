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

template< bool Symmetric,
          bool Binary >
struct MatrixType
{
   static constexpr bool isSymmetric() { return Symmetric; }

   static constexpr bool isBinary() { return Binary; }

};

struct GeneralMatrix
{
   static constexpr bool isSymmetric() { return false; }

   static constexpr bool isBinary() { return false; }
};

struct SymmetricMatrix
{
   static constexpr bool isSymmetric() { return true; }

   static constexpr bool isBinary() { return false; }
};

struct BinaryMatrix
{
   static constexpr bool isSymmetric() { return false; }

   static constexpr bool isBinary() { return true; }
};

struct BinarySymmetricMatrix
{
   static constexpr bool isSymmetric() { return false; }

   static constexpr bool isBinary() { return true; }
};

struct SymmetricBinaryMatrix
{
   static constexpr bool isSymmetric() { return false; }

   static constexpr bool isBinary() { return true; }
};


   } //namespace Matrices
} //namespace TNL