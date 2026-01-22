## fetching sample genome (Chlamydia psittaci 6BC)
#
FetchContent_Declare(
  sample_genome
  URL      https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/204/255/GCF_000204255.1_ASM20425v1/GCF_000204255.1_ASM20425v1_genomic.fna.gz
  URL_HASH MD5=599418042b774c4e200406d290f4a863
  DOWNLOAD_NO_EXTRACT TRUE
)
FetchContent_MakeAvailable(sample_genome)

## fetching sample reads (Experiment: SRX6910578)
#
FetchContent_Declare(
  sample_reads
  URL       ftp://ftp.sra.ebi.ac.uk/vol1/fastq/SRR101/073/SRR10190173/SRR10190173_1.fastq.gz
  URL_HASH MD5=f93602650632e6d0a32201457b4a2385
  DOWNLOAD_NO_EXTRACT TRUE
)
FetchContent_MakeAvailable(sample_reads)

## add path for unit tests
#
add_definitions(-DSAMPLE_GENOME="${sample_genome_SOURCE_DIR}/GCF_000204255.1_ASM20425v1_genomic.fna.gz")
add_definitions(-DSAMPLE_READS="${sample_reads_SOURCE_DIR}/SRR10190173_1.fastq.gz")

## defining target for unit tests
#
add_executable(unit_tests unit_tests.cpp)

if(CUDAToolkit_FOUND)
  set_source_files_properties(unit_tests.cpp PROPERTIES LANGUAGE CUDA)
  target_compile_options(unit_tests INTERFACE
    $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>
    $<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CUDA>>:-G;-src-in-ptx>
  )
endif()

if(hip_FOUND)
  set_source_files_properties(unit_tests.cpp PROPERTIES LANGUAGE HIP)
  target_compile_options(unit_tests PRIVATE
    -Wno-unused-result
    $<$<COMPILE_LANGUAGE:HIP>:--offload-arch=gfx942:xnack+>
  )
endif()

## defining link libraries for unit tests
#
target_link_libraries(unit_tests PRIVATE
  ${PROJECT_NAME}::${PROJECT_NAME}
  Catch2::Catch2WithMain
)

## finally adding unit tests
#
catch_discover_tests(unit_tests)
