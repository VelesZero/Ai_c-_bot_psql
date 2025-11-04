cmake_minimum_required(VERSION 3.15)
project(AIQueryAgent VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Опции компиляции
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -O2")

# Поиск зависимостей
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

# Попытка найти nlohmann_json
find_package(nlohmann_json 3.2.0 QUIET)

if(NOT nlohmann_json_FOUND)
    message(STATUS "nlohmann_json not found via find_package, using include path")
    include_directories(/usr/include /usr/local/include)
endif()

# Torch (libtorch)
set(Torch_DIR "${PROJECT_SOURCE_DIR}/libtorch/share/cmake/Torch")
set(CMAKE_PREFIX_PATH "${PROJECT_SOURCE_DIR}/libtorch;${CMAKE_PREFIX_PATH}")
find_package(Torch REQUIRED)
message(STATUS "Found Torch: ${TORCH_VERSION}")

# Prefer local libtorch at runtime
set(CMAKE_BUILD_RPATH "${PROJECT_SOURCE_DIR}/libtorch/lib")
set(CMAKE_INSTALL_RPATH "${PROJECT_SOURCE_DIR}/libtorch/lib")
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

# Включение директорий
include_directories(
    ${PROJECT_SOURCE_DIR}
    ${TORCH_INCLUDE_DIRS}
)

# Исходные файлы
set(SOURCES
    src/main.cpp
    src/core/Agent.cpp
    src/core/DatabaseConnector.cpp
    src/core/QueryBuilder.cpp
    src/core/ResponseParser.cpp
    src/nlprocessor/NLProcessor.cpp
    src/config/Config.cpp
    src/utils/Logger.cpp
    # ML (Neural network) sources
    src/ml/ModelTrainer.cpp
    src/ml/Seq2SeqModel.cpp
    src/ml/Vocabulary.cpp
)

# Создание исполняемого файла
add_executable(${PROJECT_NAME} ${SOURCES})

# Линковка библиотек
if(nlohmann_json_FOUND)
    if(TARGET Torch::Torch)
        target_link_libraries(${PROJECT_NAME} PRIVATE pqxx pq nlohmann_json::nlohmann_json pthread Torch::Torch)
    else()
        target_link_libraries(${PROJECT_NAME} PRIVATE pqxx pq nlohmann_json::nlohmann_json pthread "${TORCH_LIBRARIES}")
    endif()
else()
    if(TARGET Torch::Torch)
        target_link_libraries(${PROJECT_NAME} PRIVATE pqxx pq pthread Torch::Torch)
    else()
        target_link_libraries(${PROJECT_NAME} PRIVATE pqxx pq pthread "${TORCH_LIBRARIES}")
    endif()
endif()

# Убедимся, что используем флаги Torch
target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_17)
set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
set_target_properties(${PROJECT_NAME} PROPERTIES
    BUILD_RPATH "${PROJECT_SOURCE_DIR}/libtorch/lib"
    INSTALL_RPATH "${PROJECT_SOURCE_DIR}/libtorch/lib"
)

# Таргет для обучения модели
add_executable(train_model train_model.cpp
    src/ml/ModelTrainer.cpp
    src/ml/Seq2SeqModel.cpp
    src/ml/Vocabulary.cpp
    src/utils/Logger.cpp
)

if(nlohmann_json_FOUND)
    if(TARGET Torch::Torch)
        target_link_libraries(train_model PRIVATE nlohmann_json::nlohmann_json Torch::Torch pthread)
    else()
        target_link_libraries(train_model PRIVATE nlohmann_json::nlohmann_json "${TORCH_LIBRARIES}" pthread)
    endif()
else()
    if(TARGET Torch::Torch)
        target_link_libraries(train_model PRIVATE Torch::Torch pthread)
    else()
        target_link_libraries(train_model PRIVATE "${TORCH_LIBRARIES}" pthread)
    endif()
endif()

target_compile_features(train_model PRIVATE cxx_std_17)
set_property(TARGET train_model PROPERTY CXX_STANDARD 17)
set_target_properties(${PROJECT_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set_target_properties(train_model PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set_target_properties(train_model PROPERTIES
    BUILD_RPATH "${PROJECT_SOURCE_DIR}/libtorch/lib"
    INSTALL_RPATH "${PROJECT_SOURCE_DIR}/libtorch/lib"
)

# Test model executable
add_executable(test_model test_model.cpp)
target_include_directories(test_model PRIVATE ${CMAKE_SOURCE_DIR}/src)
#target_link_libraries(test_model PRIVATE ${PROJECT_NAME} ${PostgreSQL_LIBRARIES})

if(TARGET Torch::Torch)
    target_link_libraries(test_model PRIVATE Torch::Torch pthread)
else()
    target_link_libraries(test_model PRIVATE "${TORCH_LIBRARIES}" pthread)
endif()

target_compile_features(test_model PRIVATE cxx_std_17)
set_target_properties(test_model PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set_target_properties(test_model PROPERTIES
    BUILD_RPATH "${PROJECT_SOURCE_DIR}/libtorch/lib"
    INSTALL_RPATH "${PROJECT_SOURCE_DIR}/libtorch/lib"
)

# Training executable in training_data directory
add_executable(train_model_data
    training_data/train_model.cpp
    src/ml/ModelTrainer.cpp
    src/ml/Seq2SeqModel.cpp
    src/ml/Vocabulary.cpp
    src/utils/Logger.cpp
)

target_include_directories(train_model_data PRIVATE ${CMAKE_SOURCE_DIR}/src)

if(nlohmann_json_FOUND)
    if(TARGET Torch::Torch)
        target_link_libraries(train_model_data PRIVATE nlohmann_json::nlohmann_json Torch::Torch pthread)
    else()
        target_link_libraries(train_model_data PRIVATE nlohmann_json::nlohmann_json "${TORCH_LIBRARIES}" pthread)
    endif()
else()
    if(TARGET Torch::Torch)
        target_link_libraries(train_model_data PRIVATE Torch::Torch pthread)
    else()
        target_link_libraries(train_model_data PRIVATE "${TORCH_LIBRARIES}" pthread)
    endif()
endif()

target_compile_features(train_model_data PRIVATE cxx_std_17)
set_property(TARGET train_model_data PROPERTY CXX_STANDARD 17)
set_target_properties(train_model_data PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set_target_properties(train_model_data PROPERTIES
    BUILD_RPATH "${PROJECT_SOURCE_DIR}/libtorch/lib"
    INSTALL_RPATH "${PROJECT_SOURCE_DIR}/libtorch/lib"
)

# Установка
install(TARGETS ${PROJECT_NAME} train_model test_model train_model_data DESTINATION bin)