cmake_minimum_required(VERSION 3.23)
project(resnet18_kernel_lab LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# (중요) 추가 컴파일 옵션 일절 넣지 말기
# set(CMAKE_CUDA_FLAGS ...) 같은 것도 일단 금지

# 아키텍처 기본값 (원하면 줄여도 됨: 86 정도만)
if(NOT CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 75;80;86;89;90)
endif()

find_package(CUDAToolkit REQUIRED)

# 실행파일은 build/ 아래로
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

add_subdirectory(fp32)
