INSTALL( FILES tnl-run-heat-equation-benchmark
               run-tnl-benchmark-spmv
               run-tnl-benchmark-traversers
               run-matrix-solvers-benchmark
               run-tnl-benchmark-spmv
               run-tnl-benchmark-linear-solvers
               
               DESTINATION ${TNL_TARGET_DATA_DIRECTORY}/benchmark-scripts )

INSTALL( FILES run-tnl-benchmark-spmv
               run-tnl-benchmark-traversers
         DESTINATION bin
         PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE )
