# Minimum CMake required
cmake_minimum_required(VERSION 3.5)

if(WIN32)
	if(${CMAKE_VERSION} VERSION_LESS "3.8")
		message(WARNING "Your current cmake version is ${CMAKE_VERSION} which does not support setting the toolset architecture to x64. This may cause \"compiler out of heap space\" errors when building. Consider upgrading your cmake to > 3.8 and using the flag -Thost=x64 when running cmake. Ignore this if you are on CMake GUI.")
	else()
		if(NOT CMAKE_VS_PLATFORM_TOOLSET_HOST_ARCHITECTURE OR NOT "${CMAKE_VS_PLATFORM_TOOLSET_HOST_ARCHITECTURE}" STREQUAL "x64")
			message(WARNING "Your current cmake generator is set to use 32 bit toolset architecture. This may cause \"compiler out of heap space\" errors when building. Consider using the flag -Thost=x64 when running cmake. Ignore this if you are on CMake GUI.")
		endif()
	endif()
endif()

# Project
project(tensorflow VERSION 1.12.0 LANGUAGES C CXX)

# Set C++14 as standard for the whole project
set(CMAKE_CXX_STANDARD 14)

# Gangmuk: This is added to generate compilation database file to use it in 'YouCompleteMe'
set(CMAKE_EXPORT_COMPILE_COMMANDS 1)

# Actual source is the ../../.. directory
get_filename_component(tf_contrib_source_dir ${tensorflow_SOURCE_DIR} PATH)
get_filename_component(tf_tf_source_dir ${tf_contrib_source_dir} PATH)
get_filename_component(tensorflow_source_dir ${tf_tf_source_dir} PATH)

# [CLEANUP] Not sure if this is needed (copied from Protobuf)
# CMake policies
cmake_policy(SET CMP0022 NEW)

# Options
option(tensorflow_VERBOSE "Enable for verbose output" OFF)

if(WIN32)
# BoringSSL is disabled for windows as it currently doesn't build with
# MSBuild. (Ninja is required.)
option(tensorflow_ENABLE_SSL_SUPPORT "Enable boringssl support" OFF)
else()
# BoringSSL is enabled for gRPC.
option(tensorflow_ENABLE_SSL_SUPPORT "Enable boringssl support" ON)
endif()

option(tensorflow_ENABLE_GRPC_SUPPORT "Enable gRPC support" ON)
option(tensorflow_ENABLE_HDFS_SUPPORT "Enable HDFS support" OFF)
option(tensorflow_BUILD_CC_EXAMPLE "Build the C++ tutorial example" ON)
option(tensorflow_BUILD_PYTHON_BINDINGS "Build the Python bindings" ON)
option(tensorflow_BUILD_ALL_KERNELS "Build all OpKernels" ON)
option(tensorflow_BUILD_CONTRIB_KERNELS "Build OpKernels from tensorflow/contrib/..." ON)
option(tensorflow_BUILD_CC_TESTS "Build cc unit tests " OFF)
option(tensorflow_BUILD_PYTHON_TESTS "Build python unit tests " OFF)
option(tensorflow_BUILD_MORE_PYTHON_TESTS "Build more python unit tests for contrib packages" OFF)
option(tensorflow_BUILD_SHARED_LIB "Build TensorFlow as a shared library" OFF)
option(tensorflow_OPTIMIZE_FOR_NATIVE_ARCH "Enable compiler optimizations for the native processor architecture (if available)" ON)
option(tensorflow_ENABLE_SNAPPY_SUPPORT "Enable SNAPPY compression support" ON)
option(tensorflow_DISABLE_EIGEN_FORCEINLINE "Disable forceinline, to speed up build on windows." OFF)

if (WIN32)
SET(tensorflow_WIN_CPU_SIMD_OPTIONS "/arch:AVX" CACHE STRING "Enables CPU SIMD instructions")
SET_PROPERTY(CACHE tensorflow_WIN_CPU_SIMD_OPTIONS PROPERTY STRINGS /arch:AVX) 
endif()

# SIMD, MKL and MKLDNN options
option(tensorflow_WIN_CPU_SIMD_OPTIONS "Enables CPU SIMD instructions" OFF)
option(tensorflow_ENABLE_MKL_SUPPORT "Enable Intel MKL support" OFF)
option(tensorflow_ENABLE_MKLDNN_SUPPORT "Enable Intel MKLDNN support, requires MKL enabled" OFF)


# GPU, CUDA and cuDNN options
option(tensorflow_ENABLE_GPU "Enable GPU support" OFF)

if(HAIKU)
	option(tensorflow_ENABLE_POSITION_INDEPENDENT_CODE "Enable PIE support" OFF)
else()
	option(tensorflow_ENABLE_POSITION_INDEPENDENT_CODE "Enable PIE support" ON)
endif()


if (NOT WIN32)
  # Threads: defines CMAKE_THREAD_LIBS_INIT and adds -pthread compile option
  # for targets that link ${CMAKE_THREAD_LIBS_INIT}.
  find_package (Threads REQUIRED)

  # Options for linking CUDA/CUDNN libraries
  option(tensorflow_PATH_CUDA_LIB "Additional library search path for cudnn, nccl, culibos" /usr/local/cuda/lib64/)
  option(tensorflow_CUDNN_INCLUDE "cudnn.h header install path" /usr/include/)
  if (NOT tensorflow_CUDNN_INCLUDE)
    # option's default value is OFF. Fill it with real default values
    set(tensorflow_CUDNN_INCLUDE /usr/include)
  endif (NOT tensorflow_CUDNN_INCLUDE)
  option(tensorflow_NCCL_INCLUDE "nccl.h header install path" /usr/include/)
  if (NOT tensorflow_NCCL_INCLUDE)
    # option's default value is OFF. Fill it with real default values
    set(tensorflow_NCCL_INCLUDE /usr/include)
  endif (NOT tensorflow_NCCL_INCLUDE)
  option(tensorflow_PATH_CUDNN_LIB "Override PATH_CUDA_LIB for cudnn" ${tensorflow_PATH_CUDA_LIB})
  if (NOT tensorflow_PATH_CUDNN_LIB)
    # option's default value is OFF. Fill it with real default values
    set (tensorflow_PATH_CUDNN_LIB ${tensorflow_PATH_CUDA_LIB})
  endif (NOT tensorflow_PATH_CUDNN_LIB)
  option(tensorflow_PATH_NCCL_LIB "Override PATH_CUDA_LIB for nccl" ${tensorflow_PATH_CUDA_LIB})
  if (NOT tensorflow_PATH_NCCL_LIB)
    # option's default value is OFF. Fill it with real default values
    set (tensorflow_PATH_NCCL_LIB ${tensorflow_PATH_CUDA_LIB})
  endif (NOT tensorflow_PATH_NCCL_LIB)
  option(tensorflow_CUDA_LIBRARY_PATH "Designate the default CUDA library paths" /usr/local/cuda/lib64)
  if (NOT tensorflow_CUDA_LIBRARY_PATH)
    # option's default value is OFF. Fill it with real default values
    set(tensorflow_CUDA_LIBRARY_PATH /usr/local/cuda/lib64)
  endif (NOT tensorflow_CUDA_LIBRARY_PATH)

  # Options for linking other libraries
  option(systemlib_ZLIB "Use the system installed library as shared objects instead of downloading ZLIB and statically linking to it: ZLIB" OFF)
  option(systemlib_ABSEIL_CPP "Use the system installed library as shared objects instead of downloading ABSEIL_CPP and statically linking to it: ABSEIL_CPP" OFF)

  option(systemlib_ALL "Turn on every possible systemlib_* options" OFF)
  if (systemlib_ALL)
    set (systemlib_ZLIB ON)
    set (systemlib_ABSEIL_CPP ON)
  endif (systemlib_ALL)
endif()

if (WIN32)
  set(BOOL_WIN32 ON)
else (WIN32)
  set(BOOL_WIN32 OFF)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fPIC")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
endif (WIN32)

# [CLEANUP] Remove when done
# For debugging
function(SHOW_VARIABLES)
    get_cmake_property(_variableNames VARIABLES)
    foreach (_variableName ${_variableNames})
        message(STATUS "${_variableName}=${${_variableName}}")
    endforeach()
endfunction()

# External dependencies
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/external ${PROJECT_SOURCE_DIR}/modules)

# Location where external projects will be downloaded
set (DOWNLOAD_LOCATION "${CMAKE_CURRENT_BINARY_DIR}/downloads"
     CACHE PATH "Location where external projects will be downloaded.")
mark_as_advanced(DOWNLOAD_LOCATION)

if (tensorflow_ENABLE_POSITION_INDEPENDENT_CODE)
	set(CMAKE_POSITION_INDEPENDENT_CODE ON)
else()
	set(CMAKE_POSITION_INDEPENDENT_CODE OFF)
endif()

# TODO(jart): We should make this only apply to snapfn.cc
add_definitions(-DSQLITE_OMIT_LOAD_EXTENSION)

if (tensorflow_DISABLE_EIGEN_FORCEINLINE)
  add_definitions(-DEIGEN_STRONG_INLINE=inline)
endif()

add_definitions(-DEIGEN_AVOID_STL_ARRAY)
if(WIN32)
  if(CMAKE_SIZEOF_VOID_P EQUAL 8)
      # 64 bits
      add_definitions(-DWIN64)
  elseif(CMAKE_SIZEOF_VOID_P EQUAL 4)
      # 32 bits
      # temporary fix for #18241
      add_definitions(-DEIGEN_DEFAULT_DENSE_INDEX_TYPE=std::int64_t)
  endif()
  add_definitions(-DNOMINMAX -D_WIN32_WINNT=0x0A00)
  add_definitions(-DWIN32_LEAN_AND_MEAN -DNOGDI -DPLATFORM_WINDOWS)
  add_definitions(-DTENSORFLOW_USE_EIGEN_THREADPOOL -DEIGEN_HAS_C99_MATH)
  add_definitions(-DTF_COMPILE_LIBRARY)
  add_compile_options(/bigobj /GF /MP /Gm-)
  # Suppress warnings to reduce build log size.
  add_compile_options(/wd4267 /wd4244 /wd4800 /wd4503 /wd4554 /wd4996 /wd4348 /wd4018)
  add_compile_options(/wd4099 /wd4146 /wd4267 /wd4305 /wd4307)
  add_compile_options(/wd4715 /wd4722 /wd4723 /wd4838 /wd4309 /wd4334)
  add_compile_options(/wd4003 /wd4244 /wd4267 /wd4503 /wd4506 /wd4800 /wd4996)
  # Suppress linker warnings.
  set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} /ignore:4049 /ignore:4197 /ignore:4217 /ignore:4221")
  set(CMAKE_MODULE_LINKER_FLAGS "${CMAKE_MODULE_LINKER_FLAGS} /ignore:4049 /ignore:4197 /ignore:4217 /ignore:4221")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /ignore:4049 /ignore:4197 /ignore:4217 /ignore:4221")
  set(CMAKE_CXX_FLAGS_DEBUG "/D_DEBUG /MDd /Ob2")
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /D_ITERATOR_DEBUG_LEVEL=0")
  set(CMAKE_CXX_FLAGS_MINSIZEREL "${CMAKE_CXX_FLAGS_MINSIZEREL} /D_ITERATOR_DEBUG_LEVEL=0")
  set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} /D_ITERATOR_DEBUG_LEVEL=0")

  set(compiler_flags
    CMAKE_CXX_FLAGS
    CMAKE_CXX_FLAGS_DEBUG
    CMAKE_CXX_FLAGS_RELEASE
    CMAKE_C_FLAGS
    CMAKE_C_FLAGS_DEBUG
    CMAKE_C_FLAGS_RELEASE
  )
  # No exception
  foreach(flag ${compiler_flags})
    string(REPLACE "/EHsc" "/EHs-c-" ${flag} "${${flag}}")
  endforeach()
  add_definitions(/D_HAS_EXCEPTIONS=0)
  # Suppress 'noexcept used with no exception handling mode specified' warning
  add_compile_options(/wd4577)

  # Try to avoid flaky failures due to failed generation of generate.stamp files.
  set(CMAKE_SUPPRESS_REGENERATION ON)
endif()


if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-exceptions -std=c++11")
endif()

if (tensorflow_OPTIMIZE_FOR_NATIVE_ARCH)
  include(CheckCXXCompilerFlag)
  CHECK_CXX_COMPILER_FLAG("-march=native" COMPILER_OPT_ARCH_NATIVE_SUPPORTED)
  if (COMPILER_OPT_ARCH_NATIVE_SUPPORTED)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
  endif()
endif()

include(CheckCXXCompilerFlag)

# OpenMP Support
if (WIN32)
  CHECK_CXX_COMPILER_FLAG("/openmp" MSVC_OPENMP_SUPPORT)
  if (MSVC_OPENMP_SUPPORT)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /openmp")
  endif()
else (WIN32)
  CHECK_CXX_COMPILER_FLAG("-fopenmp" GCC_OPENMP_SUPPORT)
  if (GCC_OPENMP_SUPPORT)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
  endif()
endif (WIN32)

# MSVC SIMD instructions
if (tensorflow_WIN_CPU_SIMD_OPTIONS)
  if (WIN32)
    CHECK_CXX_COMPILER_FLAG(${tensorflow_WIN_CPU_SIMD_OPTIONS} COMPILER_OPT_WIN_CPU_SIMD_SUPPORTED)
    if(COMPILER_OPT_WIN_CPU_SIMD_SUPPORTED)
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${tensorflow_WIN_CPU_SIMD_OPTIONS}")
    endif()
  endif()
endif()

# External dependencies
include(zlib)
include(gif)
include(png)
include(jpeg)
include(lmdb)
include(eigen)
include(gemmlowp)
include(jsoncpp)
include(farmhash)
include(fft2d)
include(highwayhash)
include(nsync)
include(protobuf)
include(re2)
include(cub)
include(sqlite)
include(double_conversion)
include(abseil_cpp)
if (tensorflow_BUILD_CC_TESTS)
  include(googletest)
endif()

add_definitions(${ADD_CFLAGS})
link_directories(${ADD_LINK_DIRECTORY})

set(tensorflow_EXTERNAL_LIBRARIES
    ${tensorflow_EXTERNAL_LIBRARIES}
    ${gif_STATIC_LIBRARIES}
    ${png_STATIC_LIBRARIES}
    ${jpeg_STATIC_LIBRARIES}
    ${lmdb_STATIC_LIBRARIES}
    ${jsoncpp_STATIC_LIBRARIES}
    ${farmhash_STATIC_LIBRARIES}
    ${fft2d_STATIC_LIBRARIES}
    ${highwayhash_STATIC_LIBRARIES}
    ${nsync_STATIC_LIBRARIES}
    ${protobuf_STATIC_LIBRARIES}
    ${re2_STATIC_LIBRARIES}
    ${sqlite_STATIC_LIBRARIES}
    ${double_conversion_STATIC_LIBRARIES}
)

if (systemlib_ZLIB)
  set(tensorflow_EXTERNAL_LIBRARIES ${tensorflow_EXTERNAL_LIBRARIES}
      ${ZLIB_LIBRARIES})
else (systemlib_ZLIB)
  set(tensorflow_EXTERNAL_LIBRARIES ${tensorflow_EXTERNAL_LIBRARIES}
    ${zlib_STATIC_LIBRARIES})
endif (systemlib_ZLIB)

if (systemlib_ABSEIL_CPP)
  set(tensorflow_EXTERNAL_LIBRARIES ${tensorflow_EXTERNAL_LIBRARIES}
      ${abseil_cpp_LIBRARIES})
else (systemlib_ABSEIL_CPP)
  set(tensorflow_EXTERNAL_LIBRARIES ${tensorflow_EXTERNAL_LIBRARIES}
    ${abseil_cpp_STATIC_LIBRARIES})
endif (systemlib_ABSEIL_CPP)

set(tensorflow_EXTERNAL_DEPENDENCIES
    zlib_copy_headers_to_destination
    gif_copy_headers_to_destination
    png_copy_headers_to_destination
    jpeg_copy_headers_to_destination
    lmdb_copy_headers_to_destination
    jsoncpp
    farmhash_copy_headers_to_destination
    highwayhash_copy_headers_to_destination
    nsync_copy_headers_to_destination
    protobuf
    eigen
    gemmlowp
    cub
    fft2d
    re2
    sqlite_copy_headers_to_destination
    double_conversion
)

include_directories(
    # Source and generated code.
    ${tensorflow_source_dir}
    ${CMAKE_CURRENT_BINARY_DIR}
    # External dependencies.
    ${zlib_INCLUDE_DIR}
    ${gif_INCLUDE_DIR}
    ${png_INCLUDE_DIR}
    ${jpeg_INCLUDE_DIR}
    ${lmdb_INCLUDE_DIR}
    ${eigen_INCLUDE_DIRS}
    ${gemmlowp_INCLUDE_DIR}
    ${jsoncpp_INCLUDE_DIR}
    ${farmhash_INCLUDE_DIR}
    ${highwayhash_INCLUDE_DIR}
    ${cub_INCLUDE_DIR}
    ${nsync_INCLUDE_DIR}
    ${PROTOBUF_INCLUDE_DIRS}
    ${re2_INCLUDE_DIR}
    ${sqlite_INCLUDE_DIR}
    ${double_conversion_INCLUDE_DIR}
)

if(tensorflow_ENABLE_GRPC_SUPPORT)
  if(tensorflow_ENABLE_SSL_SUPPORT)
    include(boringssl)
    include_directories(${boringssl_INCLUDE_DIR})
  endif()
  include(grpc)
  include_directories(${GRPC_INCLUDE_DIRS})
  # Place boringssl after grpc as grpc depends on boringssl.
  list(APPEND tensorflow_EXTERNAL_LIBRARIES ${grpc_STATIC_LIBRARIES})
  list(APPEND tensorflow_EXTERNAL_DEPENDENCIES grpc)
  if(tensorflow_ENABLE_SSL_SUPPORT)
    list(APPEND tensorflow_EXTERNAL_LIBRARIES ${boringssl_STATIC_LIBRARIES})
    list(APPEND tensorflow_EXTERNAL_DEPENDENCIES boringssl)
  endif()
endif()
if(tensorflow_ENABLE_SNAPPY_SUPPORT)
  include(snappy)
  list(APPEND tensorflow_EXTERNAL_LIBRARIES ${snappy_STATIC_LIBRARIES})
  list(APPEND tensorflow_EXTERNAL_DEPENDENCIES snappy)
  include_directories(${snappy_INCLUDE_DIR})
endif()
if(WIN32)
  list(APPEND tensorflow_EXTERNAL_LIBRARIES wsock32 ws2_32 shlwapi)
endif()
if(UNIX)
  list(APPEND tensorflow_EXTERNAL_LIBRARIES ${CMAKE_THREAD_LIBS_INIT} ${CMAKE_DL_LIBS})
endif()
if(HAIKU)
  list(APPEND tensorflow_EXTERNAL_LIBRARIES network)
endif()

# MKL Support
if (tensorflow_ENABLE_MKL_SUPPORT)
  add_definitions(-DINTEL_MKL -DEIGEN_USE_VML -DENABLE_MKL)
  include(mkl)
  list(APPEND tensorflow_EXTERNAL_LIBRARIES ${mkl_STATIC_LIBRARIES})
  list(APPEND tensorflow_EXTERNAL_DEPENDENCIES mkl_copy_shared_to_destination)
  include_directories(${mkl_INCLUDE_DIRS})
  if (tensorflow_ENABLE_MKLDNN_SUPPORT)
    include(mkldnn)
    list(APPEND tensorflow_EXTERNAL_LIBRARIES ${mkldnn_STATIC_LIBRARIES})
    list(APPEND tensorflow_EXTERNAL_DEPENDENCIES mkldnn_copy_shared_to_destination)
    include_directories(${mkldnn_INCLUDE_DIRS})
  endif(tensorflow_ENABLE_MKLDNN_SUPPORT)
endif (tensorflow_ENABLE_MKL_SUPPORT)

if (tensorflow_ENABLE_GPU)
  if (NOT WIN32)
    # Default install paths for cuda libraries in Linux
    # In some Linux distros, find_package(CUDA) seems to require CMAKE_LIBRARY_PATH to include cuda-lib paths
    list(APPEND CMAKE_LIBRARY_PATH "${tensorflow_CUDA_LIBRARY_PATH}")
    list(APPEND CMAKE_LIBRARY_PATH "${tensorflow_CUDA_LIBRARY_PATH}/stubs")
  endif (NOT WIN32)

  # minimum 9.0 in cuda version
  find_package(CUDA 9.0 REQUIRED)
  if(NOT CUDA_FOUND)
    message(FATAL_ERROR "CUDA not found.")
  endif()

  # use cmake internal CUDA_ARCH_NAME switch
  # e.g. CUDA_ARCH_NAME="Auto" will autodetect
  #      CUDA_ARCH_NAME="All"  will use all arches
  cuda_select_nvcc_arch_flags(NVCC_ARCH_FLAGS ${CUDA_ARCH_NAME})
  list(APPEND CUDA_NVCC_FLAGS ${NVCC_ARCH_FLAGS})
  message(STATUS "Using CUDA arch flags: ${NVCC_ARCH_FLAGS_readable}")

  set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};--include-path ${PROJECT_BINARY_DIR}/$\{build_configuration\};--expt-relaxed-constexpr)
  set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-ftz=true)  # Flush denormals to zero
  set(CUDA_INCLUDE ${CUDA_TOOLKIT_TARGET_DIR} ${CUDA_TOOLKIT_TARGET_DIR}/extras/CUPTI/include)

  include_directories(${CUDA_INCLUDE})
  if (WIN32)
    add_definitions(-DGOOGLE_CUDA=1 -DTF_EXTRA_CUDA_CAPABILITIES=3.7,5.2,6.0,6.1,7.0)
  else (WIN32)
    # Without these double quotes, cmake in Linux makes it "-DTF_EXTRA_CUDA_CAPABILITIES=3.7, -D5.2, ..." for cc, which incurs build breaks
    add_definitions(-DGOOGLE_CUDA=1 -D"TF_EXTRA_CUDA_CAPABILITIES=3.7,5.2,6.0,6.1,7.0")
  endif (WIN32)

  if (WIN32)
    # add cudnn
    if(NOT CUDNN_HOME)
      set(CUDNN_HOME ${CUDA_TOOLKIT_TARGET_DIR})
    endif(NOT CUDNN_HOME)
    set(CUDNN_INCLUDE "${CUDNN_HOME}/include")

    set(CUDA_LIBRARIES ${CUDA_LIBRARIES} ${CUDA_CUDA_LIBRARY} ${CUDA_CUBLAS_LIBRARIES} ${CUDA_CUFFT_LIBRARIES}
      ${CUDA_curand_LIBRARY} ${CUDA_cupti_LIBRARY} ${CUDA_cusolver_LIBRARY} ${CUDNN_HOME}/lib/x64/cudnn.lib)
  else (WIN32)
    set(CUDNN_INCLUDE "${tensorflow_CUDNN_INCLUDE}")

    if (tensorflow_BUILD_SHARED_LIB)
      find_library(nccl_LIBRARY NAMES libnccl.so PATHS ${tensorflow_PATH_NCCL_LIB} ${CUDA_TOOLKIT_ROOT_DIR})
    else (tensorflow_BUILD_SHARED_LIB)
      find_library(nccl_LIBRARY NAMES libnccl_static.a PATHS ${tensorflow_PATH_NCCL_LIB} ${CUDA_TOOLKIT_ROOT_DIR})
    endif (tensorflow_BUILD_SHARED_LIB)
    if (NOT nccl_LIBRARY)
      message(FATAL_ERROR "NCCL is required for GPU-build")
    else (NOT nccl_LIBRARY)
      message("nccl: ${nccl_LIBRARY}")
      # something like /usr/lib64/libnccl_static.a
    endif (NOT nccl_LIBRARY)

    if (tensorflow_BUILD_SHARED_LIB)
      find_library(cudnn_LIBRARY NAMES libcudnn.so PATHS ${tensorflow_PATH_CUDNN_LIB} ${CUDA_TOOLKIT_ROOT_DIR})
    else (tensorflow_BUILD_SHARED_LIB)
      find_library(cudnn_LIBRARY NAMES libcudnn_static.a PATHS ${tensorflow_PATH_CUDNN_LIB} ${CUDA_TOOLKIT_ROOT_DIR})
    endif (tensorflow_BUILD_SHARED_LIB)
    if (NOT cudnn_LIBRARY)
      message(FATAL_ERROR "CUDNN is required for GPU-build")
    else (NOT cudnn_LIBRARY)
      file(READ ${CUDNN_INCLUDE}/cudnn.h CUDNN_VERSION_FILE_CONTENTS)
      # fetch cudnn version
      string(REGEX MATCH "define CUDNN_MAJOR * +([0-9]+)"
             CUDNN_VERSION_MAJOR "${CUDNN_VERSION_FILE_CONTENTS}")
      string(REGEX REPLACE "define CUDNN_MAJOR * +([0-9]+)" "\\1"
             CUDNN_VERSION_MAJOR "${CUDNN_VERSION_MAJOR}")
      string(REGEX MATCH "define CUDNN_MINOR * +([0-9]+)"
             CUDNN_VERSION_MINOR "${CUDNN_VERSION_FILE_CONTENTS}")
      string(REGEX REPLACE "define CUDNN_MINOR * +([0-9]+)" "\\1"
             CUDNN_VERSION_MINOR "${CUDNN_VERSION_MINOR}")
      string(REGEX MATCH "define CUDNN_PATCHLEVEL * +([0-9]+)"
             CUDNN_VERSION_PATCH "${CUDNN_VERSION_FILE_CONTENTS}")
      string(REGEX REPLACE "define CUDNN_PATCHLEVEL * +([0-9]+)" "\\1"
             CUDNN_VERSION_PATCH "${CUDNN_VERSION_PATCH}")
      if(NOT CUDNN_VERSION_MAJOR)
        set(CUDNN_VERSION "???")
      else()
        set(CUDNN_VERSION "${CUDNN_VERSION_MAJOR}.${CUDNN_VERSION_MINOR}.${CUDNN_VERSION_PATCH}")
      endif()
      message(STATUS "cudnn library: ${cudnn_LIBRARY} (found version: \"${CUDNN_VERSION}\")")
    endif (NOT cudnn_LIBRARY)

    if (tensorflow_BUILD_SHARED_LIB)
      # shared first (if exists) else static one
      find_library(culibos_LIBRARY NAMES libculibos.so libculibos.a PATHS ${tensorflow_PATH_CUDA_LIB} ${CUDA_TOOLKIT_ROOT_DIR})
    else (tensorflow_BUILD_SHARED_LIB)
      # only static version
      find_library(culibos_LIBRARY NAMES libculibos.a PATHS ${tensorflow_PATH_CUDA_LIB} ${CUDA_TOOLKIT_ROOT_DIR})
    endif (tensorflow_BUILD_SHARED_LIB)
    if (NOT culibos_LIBRARY)
      message(FATAL_ERROR "CULIBOS is required for GPU-build")
    else (NOT culibos_LIBRARY)
      message("culibos: ${culibos_LIBRARY}")
    endif (NOT culibos_LIBRARY)

    set(CUDA_LIBRARIES ${CUDA_LIBRARIES} ${CUDA_CUDA_LIBRARY} ${CUDA_CUBLAS_LIBRARIES} ${CUDA_CUFFT_LIBRARIES}
      ${CUDA_curand_LIBRARY} ${CUDA_cupti_LIBRARY} ${CUDA_cusolver_LIBRARY} ${cudnn_LIBRARY} ${culibos_LIBRARY} ${nccl_LIBRARY})
  endif (WIN32)
  include_directories(${CUDNN_INCLUDE})

  # Remove "." from CUDA version variable.
  string(REPLACE "." "" short_CUDA_VER ${CUDA_VERSION})

  # List of enumerated CUDA caps
  string(REPLACE " " ";" NVCC_ARCH_LIST "${NVCC_ARCH_FLAGS_readable}")
  set(list ${NVCC_ARCH_LIST})

  # Construct capability string
  foreach(NVCC_ARCH ${NVCC_ARCH_LIST})
    if (NVCC_ARCH MATCHES "sm_")
      string(REGEX REPLACE "^.sm*" "" NVCC_ARCH ${NVCC_ARCH})
      math(EXPR NVCC_ARCH_MAJOR "${NVCC_ARCH} / 10")
      math(EXPR NVCC_ARCH_MINOR "(${NVCC_ARCH} - (${NVCC_ARCH_MAJOR}*10))")
      if (TF_CUDA_CAP)
        set(TF_CUDA_CAP "${TF_CUDA_CAP},CudaVersion(\"${NVCC_ARCH_MAJOR}.${NVCC_ARCH_MINOR}\")")
      else (TF_CUDA_CAP)
        set(TF_CUDA_CAP "CudaVersion(\"${NVCC_ARCH_MAJOR}.${NVCC_ARCH_MINOR}\")")
      endif (TF_CUDA_CAP)
    endif()
  endforeach()

  # create cuda_config.h
  FILE(WRITE ${tensorflow_source_dir}/third_party/gpus/cuda/cuda_config.h
    "#ifndef CUDA_CUDA_CONFIG_H_\n"
    "#define CUDA_CUDA_CONFIG_H_\n"
    "#define TF_CUDA_CAPABILITIES ${TF_CUDA_CAP}\n"
    "#define TF_CUDA_VERSION \"64_${short_CUDA_VER}\"\n"
    "#define TF_CUDNN_VERSION \"64_${CUDNN_VERSION}\"\n"
    "#define TF_CUDA_TOOLKIT_PATH \"${CUDA_TOOLKIT_ROOT_DIR}\"\n"
    "#endif  // CUDA_CUDA_CONFIG_H_\n"
  )

  # tf assumes in various places header files to be in cuda/include. On windows the cuda sdk
  # installs them under cuda/version/include and to avoid that we need to change tf we copy a
  # few files to cuda/include
  FILE(COPY
    ${CUDA_TOOLKIT_TARGET_DIR}/include/cuda.h
    ${CUDA_TOOLKIT_TARGET_DIR}/include/cuComplex.h
    ${CUDA_TOOLKIT_TARGET_DIR}/include/cublas_v2.h
    ${CUDA_TOOLKIT_TARGET_DIR}/include/cusolverDn.h
    ${CUDA_TOOLKIT_TARGET_DIR}/include/device_functions.h
    ${CUDA_TOOLKIT_TARGET_DIR}/include/cufft.h
    ${CUDA_TOOLKIT_TARGET_DIR}/include/curand.h
    ${CUDA_TOOLKIT_TARGET_DIR}/include/cuda_runtime_api.h
    ${CUDNN_INCLUDE}/cudnn.h
    DESTINATION ${tensorflow_source_dir}/third_party/gpus/cuda/include
  )

  include_directories(${tensorflow_source_dir}/third_party/gpus)
  # add cuda libraries to tensorflow_EXTERNAL_LIBRARIES
  list(APPEND tensorflow_EXTERNAL_LIBRARIES ${CUDA_LIBRARIES})
  if(NOT WIN32)
    # add gomp to tensorflow_EXTERNAL_LIBRARIES, needed by libcusolver.so
    list(APPEND tensorflow_EXTERNAL_LIBRARIES gomp)
  endif()

  # NOTE(mrry): Update these flags when the version of CUDA or cuDNN used
  # in the default build is upgraded.
  if(WIN32)
    set(tensorflow_BUILD_INFO_FLAGS --build_config cuda --key_value
      msvcp_dll_name=msvcp140.dll
      cudart_dll_name=cudart64_${short_CUDA_VER}.dll
      cuda_version_number=${CUDA_VERSION}
      nvcuda_dll_name=nvcuda.dll
      cudnn_dll_name=cudnn64_${tensorflow_CUDNN_VERSION}.dll
      cudnn_version_number=${tensorflow_CUDNN_VERSION})
  else(WIN32)
    set(tensorflow_BUILD_INFO_FLAGS --build_config cuda --key_value
      cuda_version_number=${CUDA_VERSION}
      cudnn_version_number=${tensorflow_CUDNN_VERSION})
  endif(WIN32)
else(tensorflow_ENABLE_GPU)
  if(WIN32)
    set(tensorflow_BUILD_INFO_FLAGS --build_config cpu --key_value
      msvcp_dll_name=msvcp140.dll)
  else()
    set(tensorflow_BUILD_INFO_FLAGS --build_config cpu)
  endif()
endif(tensorflow_ENABLE_GPU)

if(tensorflow_BUILD_PYTHON_BINDINGS)
  # Find python executable
  include(FindPythonInterp)
  if(NOT ${PYTHONINTERP_FOUND})
      message(FATAL_ERROR "CMake was unable to find a python interpreter.")
  endif()
endif()

# Let's get to work!
include(tf_core_framework.cmake)
if (tensorflow_ENABLE_GPU)
  include(tf_stream_executor.cmake)
endif()

include(tf_core_cpu.cmake)
include(tf_core_ops.cmake)
include(tf_core_direct_session.cmake)
include(tf_core_kernels.cmake)
if(tensorflow_ENABLE_GRPC_SUPPORT)
  include(tf_core_distributed_runtime.cmake)
endif()
# We include tf_cc_ops first, because tf_c depends on tf_cc.
include(tf_cc_ops.cmake)
include(tf_c.cmake)
include(tf_grappler.cmake)
include(tf_core_profiler.cmake)
include(tf_core_eager_runtime.cmake)
if(tensorflow_BUILD_CC_EXAMPLE)
  include(tf_tutorials.cmake)
  include(tf_label_image_example.cmake)
endif()
include(tf_tools.cmake)
if(tensorflow_BUILD_PYTHON_BINDINGS)
  include(tf_python.cmake)
endif()
if(tensorflow_BUILD_SHARED_LIB)
  include(tf_shared_lib.cmake)
endif()
if(tensorflow_BUILD_CC_TESTS OR tensorflow_BUILD_PYTHON_TESTS)
  include(tf_tests.cmake)
endif()
