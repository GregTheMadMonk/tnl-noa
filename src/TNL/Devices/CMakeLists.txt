set (headers Cuda.h
             Cuda_impl.h
             CudaCallable.h
             CudaDeviceInfo.h
             CudaDeviceInfo_impl.h
             Host.h
             MIC.h
             SystemInfo.h
             SystemInfo_impl.h
)

INSTALL( FILES ${headers} DESTINATION ${TNL_TARGET_INCLUDE_DIRECTORY}/Devices )
