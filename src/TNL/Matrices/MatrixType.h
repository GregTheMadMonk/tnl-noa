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

/**
 * \brief Structure for specifying type of sparse matrix.
 * 
 * It is used for specification of \ref SparseMatrix type.
 */
template< bool Symmetric,
          bool Binary >
struct MatrixType
{
   static constexpr bool isSymmetric() { return Symmetric; }

   static constexpr bool isBinary() { return Binary; }

   static String getSerializationType() {
      String type;
      if( ! isBinary() && ! isSymmetric() )
         type = "General";
      else
      {
         if( isSymmetric ) type = "Symmetric";
         if( isBinary ) type += "Binary";
      }
      return type;
   }
};

/**
 * \brief General non-symmetric matrix type.
 * 
 * It is used for specification of \ref SparseMatrix type.
 */
struct GeneralMatrix : MatrixType< false, false > {};

/**
 * \brief Symmetric matrix type.
 * 
 * Symmetric matrix stores only lower part of the matrix and its diagonal. The
 * upper part is reconstructed on the fly.
 * It is used for specification of \ref SparseMatrix type.
 */
struct SymmetricMatrix : MatrixType< true, false > {};

/**
 * \brief Binary matrix type.
 * 
 * Binary matrix does not store explictly values of matrix elements and thus
 * it reduces memory consumption.
 * It is used for specification of \ref SparseMatrix type. 
 */
struct BinaryMatrix : MatrixType< false, true > {};

/**
 * \brief Symmetric and binary matrix type.
 * 
 * Symmetric matrix stores only lower part of the matrix and its diagonal. The
 * upper part is reconstructed on the fly.
 * Binary matrix does not store explictly values of matrix elements and thus
 * it reduces memory consumption.
 * It is used for specification of \ref SparseMatrix type.
 */
struct BinarySymmetricMatrix : MatrixType< true, true > {};

/**
 * \brief Symmetric and binary matrix type.
 * 
 * Symmetric matrix stores only lower part of the matrix and its diagonal. The
 * upper part is reconstructed on the fly.
 * Binary matrix does not store explictly values of matrix elements and thus
 * it reduces memory consumption.
 * It is used for specification of \ref SparseMatrix type.
 */
struct SymmetricBinaryMatrix : MatrixType< true, true > {};

} // namespace Matrices
} // namespace TNL
