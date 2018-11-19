SET( headers DistributedArray.h
             DistributedArray_impl.h
             DistributedArrayView.h
             DistributedArrayView_impl.h
             DistributedMatrix.h
             DistributedMatrix_impl.h
             DistributedSpMV.h
             DistributedVector.h
             DistributedVector_impl.h
             DistributedVectorView.h
             DistributedVectorView_impl.h
             Partitioner.h
             Subrange.h
    )

INSTALL( FILES ${headers} DESTINATION ${TNL_TARGET_INCLUDE_DIRECTORY}/DistributedContainers )
