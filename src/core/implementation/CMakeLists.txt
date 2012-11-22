SET( headers cuda-long-vector-kernels.h
             vector-operations.h
             memory-functions.h
             tnlArray_impl.h
             tnlMultiArray1D_impl.h
             tnlMultiArray2D_impl.h
             tnlMultiArray3D_impl.h
             tnlMultiArray4D_impl.h
             tnlMultiVector1D_impl.h
             tnlMultiVector2D_impl.h
             tnlMultiVector3D_impl.h
             tnlMultiVector4D_impl.h            
             tnlSharedArray_impl.h
             tnlSharedVector_impl.h
             tnlVector_impl.h )

INSTALL( FILES ${headers} DESTINATION include/tnl-${tnlVersion}/core/implementation )

