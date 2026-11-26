cmake_minimum_required(VERSION 3.15)
project(HNSW LANGUAGES CXX)

# ---------------------------------------------------------------------------
# HNSW 的 Python 扩展（模块名 hnsw_cpp）
#
# 注意：PATH 上的 cmake 若是 3.18，不认识 "Visual Studio 17 2022" 生成器，
# 直接用 VS 自带的 cmake（3.31）：
#
#   $cmake = "D:/VisualStudio/Common7/IDE/CommonExtensions/Microsoft/CMake/CMake/bin/cmake.exe"
#   & $cmake -S . -B build -G "Visual Studio 17 2022" -A x64 `
#            -DPython3_EXECUTABLE=D:/miniconda3/envs/cuda13_4/python.exe
#   & $cmake --build build --config Release
#
# 生成物直接落到 src/ANN/hnsw_cpp.cp313-win_amd64.pyd，
# 所以 `from . import hnsw_cpp` 立刻可用，不需要再 copy。
# ---------------------------------------------------------------------------

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "build type" FORCE)
endif()

set(HNSW_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src/ANN")
set(HNSW_INCLUDE_DIR "${HNSW_SOURCE_DIR}")
set(HNSW_PYTHON_OUT_DIR "${HNSW_SOURCE_DIR}")

set(HNSW_WARN_FLAGS "")
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    set(HNSW_WARN_FLAGS -Wall -Wextra)
endif()

# ---------- 核心实现 ----------
# HNSW.hpp 是 header-only；若以后加了 HNSW.cpp 就编成静态库，否则只做接口库。
set(HNSW_CORE_SOURCE "${HNSW_SOURCE_DIR}/HNSW.cpp")
if(EXISTS "${HNSW_CORE_SOURCE}")
    add_library(hnsw_core STATIC "${HNSW_CORE_SOURCE}")
    target_include_directories(hnsw_core PUBLIC "${HNSW_INCLUDE_DIR}")
    target_compile_options(hnsw_core PRIVATE ${HNSW_WARN_FLAGS})
else()
    message(STATUS "hnsw: 没有 ${HNSW_CORE_SOURCE}，HNSW.hpp 按 header-only 使用")
    add_library(hnsw_core INTERFACE)
    target_include_directories(hnsw_core INTERFACE "${HNSW_INCLUDE_DIR}")
endif()

# ---------- Python 扩展模块 ----------
set(HNSW_BINDING_SOURCE "${HNSW_SOURCE_DIR}/bindings.cpp"
    CACHE FILEPATH "pybind11 绑定源文件")

if(EXISTS "${HNSW_BINDING_SOURCE}")
    if(WIN32 AND NOT MSVC)
        # torch 官方 wheel 是 MSVC ABI，用 MinGW 编出来的 .pyd 链接/加载都会炸
        message(FATAL_ERROR
            "Windows 上必须用 MSVC 构建（-G \"Visual Studio 17 2022\" 或 Ninja + vcvars64）。"
            "当前编译器：${CMAKE_CXX_COMPILER_ID}")
    endif()

    find_package(Python3 COMPONENTS Interpreter Development.Module REQUIRED)

    # pybind11 装在 site-packages 里时 CMake 默认找不到它的 cmake 目录
    if(NOT pybind11_DIR)
        execute_process(
            COMMAND "${Python3_EXECUTABLE}" -m pybind11 --cmakedir
            OUTPUT_VARIABLE _pybind11_cmake_dir
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET)
        if(_pybind11_cmake_dir)
            set(pybind11_DIR "${_pybind11_cmake_dir}"
                CACHE PATH "pybind11 CMake config dir" FORCE)
        endif()
    endif()
    find_package(pybind11 CONFIG REQUIRED)

    # ---------- torch ----------
    # bindings.cpp 用了 at::Tensor，include 路径和 C++ 库都得从 torch 里取
    execute_process(
        COMMAND "${Python3_EXECUTABLE}" -c
"import os, torch
from torch.utils import cpp_extension as ce
print('|'.join(ce.include_paths()))
print(os.path.join(os.path.dirname(torch.__file__), 'lib'))"
        OUTPUT_VARIABLE _torch_info
        ERROR_VARIABLE _torch_error
        RESULT_VARIABLE _torch_result
        OUTPUT_STRIP_TRAILING_WHITESPACE)

    if(NOT _torch_result EQUAL 0)
        message(FATAL_ERROR
            "在 ${Python3_EXECUTABLE} 里 import torch 失败：\n${_torch_error}\n"
            "请用 -DPython3_EXECUTABLE=<装了 torch 的解释器> 重新配置。")
    endif()

    string(REPLACE "\n" ";" _torch_lines "${_torch_info}")
    list(GET _torch_lines 0 _torch_include_line)
    list(GET _torch_lines 1 _torch_lib_dir)
    string(REPLACE "|" ";" _torch_include_dirs "${_torch_include_line}")

    pybind11_add_module(hnsw_cpp MODULE "${HNSW_BINDING_SOURCE}")
    target_link_libraries(hnsw_cpp PRIVATE hnsw_core)
    target_include_directories(hnsw_cpp PRIVATE "${HNSW_INCLUDE_DIR}" ${_torch_include_dirs})
    target_link_directories(hnsw_cpp PRIVATE "${_torch_lib_dir}")
    # at::Tensor 的 pybind11 type_caster 定义在 torch_python 里，不链就 LNK2001；
    # 运行时还要先 import torch（它才会把 torch/lib 加进 DLL 搜索目录）。
    target_link_libraries(hnsw_cpp PRIVATE torch_python torch torch_cpu c10)
    target_compile_options(hnsw_cpp PRIVATE ${HNSW_WARN_FLAGS})

    if(MSVC)
        # /utf-8：源码是 UTF-8（含中文注释），否则 MSVC 按 GBK 代码页解析会报 C4819
        target_compile_options(hnsw_cpp PRIVATE /O2 /bigobj /utf-8 /wd4251 /wd4275)
    endif()
    if(MINGW)
        # 让 .pyd 自带 MinGW 运行时（libstdc++/libgcc/libwinpthread）
        target_link_options(hnsw_cpp PRIVATE -static)
    endif()

    # 输出直接放进 Python 包目录；多配置生成器（VS）要给每个配置都覆盖，
    # 否则会落到 build/Release/ 下面
    set_target_properties(hnsw_cpp PROPERTIES
        PREFIX ""
        LIBRARY_OUTPUT_DIRECTORY "${HNSW_PYTHON_OUT_DIR}"
        RUNTIME_OUTPUT_DIRECTORY "${HNSW_PYTHON_OUT_DIR}")
    foreach(_cfg DEBUG RELEASE RELWITHDEBINFO MINSIZEREL)
        set_target_properties(hnsw_cpp PROPERTIES
            LIBRARY_OUTPUT_DIRECTORY_${_cfg} "${HNSW_PYTHON_OUT_DIR}"
            RUNTIME_OUTPUT_DIRECTORY_${_cfg} "${HNSW_PYTHON_OUT_DIR}")
    endforeach()

    message(STATUS "hnsw_cpp: Python ${Python3_VERSION}, torch lib=${_torch_lib_dir}")
    message(STATUS "hnsw_cpp: 输出到 ${HNSW_PYTHON_OUT_DIR}")
else()
    message(STATUS "hnsw: ${HNSW_BINDING_SOURCE} 不存在，只构建 hnsw_core")
endif()