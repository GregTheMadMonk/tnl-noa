INSTALL( FILES matrix-market
               florida-matrix-market
               get-matrices
               convert-matrices
               draw-matrices
               run-matrix-solvers-benchmark
               run-tnl-benchmark-spmv
               run-tnl-benchmark-linear-solvers
               tnl-run-heat-equation-benchmark
               cuda-profiler.conf
               process-cuda-profile.pl 
               DESTINATION ${TNL_TARGET_DATA_DIRECTORY}/benchmark-scripts )

INSTALL( FILES tnl-run-spmv-benchmark
         DESTINATION bin
         PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE )
