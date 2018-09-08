ADD_SUBDIRECTORY( Algorithms )
ADD_SUBDIRECTORY( Multimaps )

set( headers Array.h
             Array_impl.h
             ArrayIO.h
             IndexedMap.h
             IndexedMap_impl.h
             IndexedSet.h
             IndexedSet_impl.h
             UnorderedIndexedSet.h
             UnorderedIndexedSet_impl.h
             List.h
             List_impl.h
             MultiArray.h
             MultiArray1D_impl.h
             MultiArray2D_impl.h
             MultiArray3D_impl.h
             MultiArray4D_impl.h
             ArrayView.h
             ArrayView_impl.h
             StaticArray.h
             StaticArray_impl.h
             StaticArray1D_impl.h
             StaticArray2D_impl.h
             StaticArray3D_impl.h
             Vector.h
             Vector_impl.h
             MultiVector.h
             MultiVector1D_impl.h
             MultiVector2D_impl.h
             MultiVector3D_impl.h
             MultiVector4D_impl.h
             VectorView.h
             VectorView_impl.h
             StaticVector.h
             StaticVector_impl.h
             StaticVector1D_impl.h
             StaticVector2D_impl.h
             StaticVector3D_impl.h  )

INSTALL( FILES ${headers} DESTINATION ${TNL_TARGET_INCLUDE_DIRECTORY}/Containers )
