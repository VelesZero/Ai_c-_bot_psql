cmake_minimum_required(VERSION 3.15)
project(AIQueryAgent VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Опции компиляции
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -O3")

# ─── Зависимости ───

# PostgreSQL
find_package(PostgreSQL QUIET)
if(NOT PostgreSQL_FOUND)
    message(WARNING "PostgreSQL not found via find_package, trying pg_config fallback")
    find_program(PG_CONFIG_EXECUTABLE NAMES pg_config)
    if(PG_CONFIG_EXECUTABLE)
        execute_process(COMMAND ${PG_CONFIG_EXECUTABLE} --includedir
                        OUTPUT_VARIABLE PG_INCLUDE_DIR
                        OUTPUT_STRIP_TRAILING_WHITESPACE)
        execute_process(COMMAND ${PG_CONFIG_EXECUTABLE} --libdir
                        OUTPUT_VARIABLE PG_LIB_DIR
                        OUTPUT_STRIP_TRAILING_WHITESPACE)
        if(PG_INCLUDE_DIR)
            message(STATUS "Using PostgreSQL include from pg_config: ${PG_INCLUDE_DIR}")
            include_directories(${PG_INCLUDE_DIR})
        endif()
        if(PG_LIB_DIR)
            message(STATUS "Using PostgreSQL libdir from pg_config: ${PG_LIB_DIR}")
            link_directories(${PG_LIB_DIR})
        endif()
    else()
        message(FATAL_ERROR "pg_config not found and find_package(PostgreSQL) failed")
    endif()
else()
    include_directories(${PostgreSQL_INCLUDE_DIRS})
endif()

# nlohmann_json
find_package(nlohmann_json 3.2.0 QUIET)
if(NOT nlohmann_json_FOUND)
    message(STATUS "nlohmann_json not found via find_package, using include path")
    include_directories(/usr/include /usr/local/include)
endif()

# Torch (libtorch)
set(Torch_DIR "${PROJECT_SOURCE_DIR}/libtorch/share/cmake/Torch")
set(CMAKE_PREFIX_PATH "${PROJECT_SOURCE_DIR}/libtorch;${CMAKE_PREFIX_PATH}")

if(NOT TARGET CUDA::nvToolsExt AND EXISTS "${PROJECT_SOURCE_DIR}/libtorch/lib/libnvToolsExt.so")
    add_library(CUDA::nvToolsExt SHARED IMPORTED)
    set_target_properties(CUDA::nvToolsExt PROPERTIES
        IMPORTED_LOCATION "${PROJECT_SOURCE_DIR}/libtorch/lib/libnvToolsExt.so"
    )
endif()

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
message(STATUS "Found Torch: ${TORCH_VERSION}")

# Runtime library paths
set(CMAKE_BUILD_RPATH "${PROJECT_SOURCE_DIR}/libtorch/lib")
set(CMAKE_INSTALL_RPATH "${PROJECT_SOURCE_DIR}/libtorch/lib")
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

include_directories(
    ${PROJECT_SOURCE_DIR}
    ${TORCH_INCLUDE_DIRS}
)

# ─── Helper: collect common link libraries ───

set(TORCH_LINK_TARGET "")
if(TARGET Torch::Torch)
    set(TORCH_LINK_TARGET Torch::Torch)
else()
    set(TORCH_LINK_TARGET "${TORCH_LIBRARIES}")
endif()

set(JSON_LINK_TARGET "")
if(nlohmann_json_FOUND)
    set(JSON_LINK_TARGET nlohmann_json::nlohmann_json)
endif()

# ─── ML sources (shared between main app and train_model) ───

set(ML_SOURCES
    src/ml/ModelTrainer.cpp
    src/ml/Seq2SeqModel.cpp
    src/ml/Vocabulary.cpp
    src/utils/Logger.cpp
)

# ─── Main executable ───

set(SOURCES
    src/main.cpp
    src/core/Agent.cpp
    src/core/DatabaseConnector.cpp
    src/core/QueryBuilder.cpp
    src/core/ResponseParser.cpp
    src/nlprocessor/NLProcessor.cpp
    src/config/Config.cpp
    ${ML_SOURCES}
)

add_executable(${PROJECT_NAME} ${SOURCES})
target_link_libraries(${PROJECT_NAME} PRIVATE pqxx pq ${JSON_LINK_TARGET} pthread ${TORCH_LINK_TARGET})
set_target_properties(${PROJECT_NAME} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    BUILD_RPATH "${PROJECT_SOURCE_DIR}/libtorch/lib"
    INSTALL_RPATH "${PROJECT_SOURCE_DIR}/libtorch/lib"
)

# ─── Training executable ───

add_executable(train_model train_model.cpp ${ML_SOURCES})
target_link_libraries(train_model PRIVATE ${JSON_LINK_TARGET} ${TORCH_LINK_TARGET} pthread)
set_target_properties(train_model PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    BUILD_RPATH "${PROJECT_SOURCE_DIR}/libtorch/lib"
    INSTALL_RPATH "${PROJECT_SOURCE_DIR}/libtorch/lib"
)

# ─── Test model executable ───

add_executable(test_model test_model.cpp)
target_include_directories(test_model PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(test_model PRIVATE ${TORCH_LINK_TARGET} pthread)
set_target_properties(test_model PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    BUILD_RPATH "${PROJECT_SOURCE_DIR}/libtorch/lib"
    INSTALL_RPATH "${PROJECT_SOURCE_DIR}/libtorch/lib"
)

# ─── Dataset generator ───

add_executable(generate_bookings_dataset_from_db
    training_data/generate_bookings_dataset_from_db.cpp
)
target_link_libraries(generate_bookings_dataset_from_db PRIVATE pqxx pq ${JSON_LINK_TARGET} pthread)
set_target_properties(generate_bookings_dataset_from_db PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# ─── Install ───

install(TARGETS ${PROJECT_NAME} train_model test_model generate_bookings_dataset_from_db DESTINATION bin)
