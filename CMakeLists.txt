cmake_minimum_required(VERSION 3.20)
project(qwen38 LANGUAGES C CXX)

# This build adds text tokenization.  The plain `make` / one-file C++ build
# deliberately remains available as the dependency-free numerical reference.
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(QWEN38_ENABLE_TOKENIZER "Build qwen38 with the Hugging Face tokenizer adapter" ON)
if(QWEN38_ENABLE_TOKENIZER)
    include(ExternalProject)
    find_package(Threads REQUIRED)
    find_program(QWEN38_CARGO_EXECUTABLE NAMES cargo
        HINTS "$ENV{CARGO_HOME}/bin" "$ENV{HOME}/.cargo/bin"
        REQUIRED)

    # Pinned source revision. We compile only rust/tokenizers-c (the Hugging
    # Face tokenizer C ABI), not tokenizers-cpp's SentencePiece wrapper.
    set(QWEN38_TOKENIZERS_SOURCE_DIR "${CMAKE_BINARY_DIR}/_deps/tokenizers-cpp-src")
    set(QWEN38_TOKENIZERS_CARGO_DIR "${CMAKE_BINARY_DIR}/tokenizers-cargo")
    set(QWEN38_TOKENIZERS_LIBRARY "${QWEN38_TOKENIZERS_CARGO_DIR}/release/libtokenizers_c.a")
    ExternalProject_Add(qwen38_tokenizers
        GIT_REPOSITORY https://github.com/mlc-ai/tokenizers-cpp.git
        GIT_TAG c586c52f93f7b060753bd2388eb96a105cb7374d
        GIT_SHALLOW TRUE
        GIT_SUBMODULES ""
        UPDATE_DISCONNECTED TRUE
        SOURCE_DIR "${QWEN38_TOKENIZERS_SOURCE_DIR}"
        CONFIGURE_COMMAND ""
        BUILD_COMMAND "${CMAKE_COMMAND}" -E chdir "${QWEN38_TOKENIZERS_SOURCE_DIR}/rust"
                      "${CMAKE_COMMAND}" -E env "MAKEFLAGS=" "CARGO_TARGET_DIR=${QWEN38_TOKENIZERS_CARGO_DIR}"
                      "${QWEN38_CARGO_EXECUTABLE}" build --release
        BUILD_IN_SOURCE TRUE
        INSTALL_COMMAND ""
        BUILD_BYPRODUCTS "${QWEN38_TOKENIZERS_LIBRARY}")

    add_executable(qwen38 qwen38.cpp qwen38_tokenizer.cpp)
    add_dependencies(qwen38 qwen38_tokenizers)
    target_compile_definitions(qwen38 PRIVATE QWEN38_WITH_TOKENIZER=1)
    target_include_directories(qwen38 PRIVATE "${QWEN38_TOKENIZERS_SOURCE_DIR}/include")
    target_link_libraries(qwen38 PRIVATE "${QWEN38_TOKENIZERS_LIBRARY}" ${CMAKE_DL_LIBS} Threads::Threads)

    # The daily real-weight development checkpoint. This remains a fixed
    # Qwen3.5-0.8B build, not a runtime-selectable model abstraction.
    add_executable(qwen38_08b qwen38.cpp qwen38_tokenizer.cpp)
    add_dependencies(qwen38_08b qwen38_tokenizers)
    target_compile_definitions(qwen38_08b PRIVATE QWEN38_WITH_TOKENIZER=1 QWEN38_DEV_08B_MODEL=1)
    target_include_directories(qwen38_08b PRIVATE "${QWEN38_TOKENIZERS_SOURCE_DIR}/include")
    target_link_libraries(qwen38_08b PRIVATE "${QWEN38_TOKENIZERS_LIBRARY}" ${CMAKE_DL_LIBS} Threads::Threads)
else()
    add_executable(qwen38 qwen38.cpp)
    add_executable(qwen38_08b qwen38.cpp)
    target_compile_definitions(qwen38_08b PRIVATE QWEN38_DEV_08B_MODEL=1)
endif()
target_compile_options(qwen38 PRIVATE -O3 -Wall -Wextra -Wpedantic)
target_compile_options(qwen38_08b PRIVATE -O3 -Wall -Wextra -Wpedantic)

# A 4-layer, width-32 Qwen3.8-shaped executable. It uses precisely the same
# loader and forward code and can create a 94 KiB fake BF16 checkpoint.
add_executable(qwen38_tiny qwen38.cpp)
target_compile_definitions(qwen38_tiny PRIVATE QWEN38_TINY_MODEL=1)
target_compile_options(qwen38_tiny PRIVATE -O3 -Wall -Wextra -Wpedantic)

option(QWEN38_ENABLE_CUDA "Build CUDA correctness backends" ON)
if(QWEN38_ENABLE_CUDA)
    include(CheckLanguage)
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
        enable_language(CUDA)
        add_executable(qwen38_tiny_cuda kernels/qwen38_tiny_cuda.cu)
        set_target_properties(qwen38_tiny_cuda PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON)
        target_compile_options(qwen38_tiny_cuda PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3>)

        # A deliberately direct, full-real-weight CUDA correctness path for
        # the daily Qwen3.5-0.8B checkpoint. It is not performance-oriented.
        add_executable(qwen38_08b_cuda kernels/qwen38_08b_cuda.cu)
        set_target_properties(qwen38_08b_cuda PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON)
        target_compile_options(qwen38_08b_cuda PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3>)
    else()
        message(STATUS "CUDA compiler not found: skipping qwen38_tiny_cuda")
    endif()
endif()
