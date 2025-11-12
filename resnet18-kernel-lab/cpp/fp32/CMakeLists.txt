# 실행파일을 build/fp32/ 아래에 두기
set(FP32_OUTDIR ${CMAKE_BINARY_DIR}/fp32)

set(FP32_KERNELS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/kernels)
set(FP32_RUNTIME_DIR ${CMAKE_CURRENT_SOURCE_DIR}/runtime)

set(FP32_INCLUDES
  ${FP32_RUNTIME_DIR}
  ${FP32_KERNELS_DIR}
  ${CUDAToolkit_INCLUDE_DIRS}
)

# 공통 커널 목록 (실제 파일 존재해야 함)
set(KERNELS_COMMON
  ${FP32_KERNELS_DIR}/im2col.cu
  ${FP32_KERNELS_DIR}/sgemm_tiled.cu
  ${FP32_KERNELS_DIR}/bn_inference.cu
  ${FP32_KERNELS_DIR}/relu.cu
)

# ---- Step2
add_executable(step2_conv1_bn1_relu
  ${KERNELS_COMMON}
  ${FP32_RUNTIME_DIR}/infer_conv1_bn1_relu.cu
)
target_include_directories(step2_conv1_bn1_relu PRIVATE ${FP32_INCLUDES})
target_link_libraries(step2_conv1_bn1_relu PRIVATE CUDA::cudart)
set_target_properties(step2_conv1_bn1_relu PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  RUNTIME_OUTPUT_DIRECTORY ${FP32_OUTDIR}
)

# ---- Step3
add_executable(step3_layer1
  ${KERNELS_COMMON}
  ${FP32_KERNELS_DIR}/add.cu
  ${FP32_RUNTIME_DIR}/infer_layer1.cu
)
target_include_directories(step3_layer1 PRIVATE ${FP32_INCLUDES})
target_link_libraries(step3_layer1 PRIVATE CUDA::cudart)
set_target_properties(step3_layer1 PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  RUNTIME_OUTPUT_DIRECTORY ${FP32_OUTDIR}
)

add_executable(step4_layer2
  runtime/infer_layer2.cu
  kernels/im2col.cu
  kernels/sgemm_tiled.cu
  kernels/bn_inference.cu
  kernels/relu.cu
  kernels/add.cu
)
target_include_directories(step4_layer2 PRIVATE ${CMAKE_CURRENT_LIST_DIR}/runtime)
target_link_libraries(step4_layer2 PRIVATE CUDA::cudart)
set_target_properties(step4_layer2 PROPERTIES CUDA_ARCHITECTURES "native")


add_executable(step5_layer3
  kernels/im2col.cu
  kernels/sgemm_tiled.cu
  kernels/bn_inference.cu
  kernels/relu.cu
  kernels/add.cu
  runtime/infer_layer3.cu
)
target_include_directories(step5_layer3 PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
target_link_libraries(step5_layer3 PRIVATE cudart)
set_target_properties(step5_layer3 PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# cpp/fp32/CMakeLists.txt (추가 블록)
add_executable(step6_layer4
    kernels/im2col.cu
    kernels/sgemm_tiled.cu
    kernels/bn_inference.cu
    kernels/relu.cu
    kernels/add.cu
    runtime/infer_layer4.cu
)
target_include_directories(step6_layer4 PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/runtime)
target_link_libraries(step6_layer4 PRIVATE CUDA::cudart)
set_target_properties(step6_layer4 PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# ... (기존 내용 유지)

add_executable(step7_head
    kernels/im2col.cu          # 이미 사용 중이면 그대로
    kernels/sgemm_tiled.cu
    kernels/relu.cu            # 필요없어도 링크 편의상 유지 가능
    kernels/bn_inference.cu    # 필요없어도 링크 편의상 유지 가능
    kernels/gap_global.cu      # <-- 신규
    kernels/softmax.cu         # <-- 신규
    runtime/infer_head.cu      # <-- 신규
)
target_include_directories(step7_head PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
target_link_libraries(step7_head PRIVATE CUDA::cudart)
set_target_properties(step7_head PROPERTIES CUDA_SEPARABLE_COMPILATION ON)


# Step8 E2E
add_executable(step8_e2e
    runtime/infer_e2e.cu
    kernels/im2col.cu
    kernels/sgemm_tiled.cu
    kernels/bn_inference.cu
    kernels/relu.cu
    kernels/add.cu
    kernels/gap_global.cu    
    kernels/maxpool2d.cu
)

target_include_directories(step8_e2e PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
target_link_libraries(step8_e2e PRIVATE CUDA::cudart)
set_target_properties(step8_e2e PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
