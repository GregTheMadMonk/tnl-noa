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

struct GeneralMatrix : MatrixType< false, false > {};

struct SymmetricMatrix : MatrixType< true, false > {};

struct BinaryMatrix : MatrixType< false, true > {};

struct BinarySymmetricMatrix : MatrixType< true, true > {};

struct SymmetricBinaryMatrix : MatrixType< true, true > {};

} // namespace Matrices
} // namespace TNL
