cmake_minimum_required(VERSION 3.15)
project(AIQueryAgent VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Опции компиляции
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -O2")

# Поиск зависимостей
find_package(PostgreSQL REQUIRED)

# Попытка найти nlohmann_json
find_package(nlohmann_json 3.2.0 QUIET)

if(NOT nlohmann_json_FOUND)
    message(STATUS "nlohmann_json not found via find_package, using include path")
    include_directories(/usr/include /usr/local/include)
endif()

# Включение директорий
include_directories(
    ${PostgreSQL_INCLUDE_DIRS}
    ${PROJECT_SOURCE_DIR}
)

# Исходные файлы
set(SOURCES
    src/main.cpp
    src/core/Agent.cpp
    src/core/DatabaseConnector.cpp
    src/core/QueryBuilder.cpp
    src/core/ResponseParser.cpp
    src/nlprocessor/NLProcessor.cpp
    src/nlprocessor/Tokenizer.cpp
    src/nlprocessor/ModelTrainer.cpp
    src/config/Config.cpp
    src/utils/Logger.cpp
)

# Создание исполняемого файла
add_executable(${PROJECT_NAME} ${SOURCES})

# Линковка библиотек
if(nlohmann_json_FOUND)
    target_link_libraries(${PROJECT_NAME}
        pqxx
        pq
        nlohmann_json::nlohmann_json
        pthread
    )
else()
    target_link_libraries(${PROJECT_NAME}
        pqxx
        pq
        pthread
    )
endif()

# Установка
install(TARGETS ${PROJECT_NAME} DESTINATION bin)
