# preset that enables GPU and selects CUDA API

set(PKG_GPU ON CACHE BOOL "Build GPU package" FORCE)
set(GPU_API "cuda" CACHE STRING "APU used by GPU package" FORCE)
set(GPU_PREC "mixed" CACHE STRING "" FORCE)
set(GPU_ARCH "sm_60" CACHE STRING "LAMMPS GPU CUDA SM primary architecture" FORCE)

