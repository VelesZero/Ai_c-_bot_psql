#!/bin/bash

echo "Fixing include paths in all source files..."

# Находим все .h и .cpp файлы
find src -type f \( -name "*.h" -o -name "*.cpp" \) | while read file; do
    echo "Processing: $file"
    
    # Заменяем пути include
    sed -i 's|#include "core/|#include "src/core/|g' "$file"
    sed -i 's|#include "nlprocessor/|#include "src/nlprocessor/|g' "$file"
    sed -i 's|#include "config/|#include "src/config/|g' "$file"
    sed -i 's|#include "utils/|#include "src/utils/|g' "$file"
    
    # Избегаем двойного src/src/
    sed -i 's|#include "src/src/|#include "src/|g' "$file"
done

echo ""
echo "✅ All includes fixed!"
echo ""
echo "Checking results..."
grep -n '#include "' src/**/*.{h,cpp} 2>/dev/null | grep -E '(core/|nlprocessor/|config/|utils/)' | head -20

