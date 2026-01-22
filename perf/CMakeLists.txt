## defining target for benchmarks
#
add_executable(benchmarks benchmarks.cpp)

if(CUDAToolkit_FOUND)
  set_source_files_properties(benchmarks.cpp PROPERTIES LANGUAGE CUDA)
  target_compile_options(benchmarks INTERFACE
    $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>
    $<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CUDA>>:-G;-src-in-ptx>
  )
endif()

if(hip_FOUND)
  set_source_files_properties(benchmarks.cpp PROPERTIES LANGUAGE HIP)
  target_compile_options(benchmarks PRIVATE
    -Wno-unused-result
    $<$<COMPILE_LANGUAGE:HIP>:--offload-arch=gfx942:xnack+>
  )
endif()

## defining link libraries for benchmarks
#
target_link_libraries(benchmarks PRIVATE
  ${PROJECT_NAME}::${PROJECT_NAME}
  benchmark::benchmark
)
