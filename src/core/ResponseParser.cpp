#include "src/core/ResponseParser.h"
#include <sstream>
#include <iomanip>
#include <algorithm>

ResponseParser::ResponseParser() {}

std::string ResponseParser::formatResponse(const DatabaseConnector::QueryResult& result,
                                          OutputFormat format) {
    if (!result.success) {
        return "Error: " + result.errorMessage;
    }
    
    switch (format) {
        case OutputFormat::JSON:
            return toJSON(result).dump(2);
        case OutputFormat::TABLE:
            return toTable(result);
        case OutputFormat::CSV:
            return toCSV(result);
        case OutputFormat::PLAIN:
            return toPlainText(result);
        default:
            return toTable(result);
    }
}

json ResponseParser::toJSON(const DatabaseConnector::QueryResult& result) {
    json output;
    output["success"] = result.success;
    output["rowCount"] = result.rowCount;
    
    if (!result.success) {
        output["error"] = result.errorMessage;
        return output;
    }
    
    json rows = json::array();
    
    for (const auto& row : result.rows) {
        json rowObj;
        for (size_t i = 0; i < result.columns.size() && i < row.size(); ++i) {
            rowObj[result.columns[i]] = row[i];
        }
        rows.push_back(rowObj);
    }
    
    output["columns"] = result.columns;
    output["data"] = rows;
    
    return output;
}

std::vector<size_t> ResponseParser::calculateColumnWidths(
    const DatabaseConnector::QueryResult& result) {
    
    std::vector<size_t> widths(result.columns.size(), 0);
    
    // Ширина заголовков
    for (size_t i = 0; i < result.columns.size(); ++i) {
        widths[i] = result.columns[i].length();
    }
    
    // Ширина данных
    for (const auto& row : result.rows) {
        for (size_t i = 0; i < row.size() && i < widths.size(); ++i) {
            widths[i] = std::max(widths[i], row[i].length());
        }
    }
    
    return widths;
}

std::string ResponseParser::padString(const std::string& str, size_t width) {
    if (str.length() >= width) return str;
    return str + std::string(width - str.length(), ' ');
}

std::string ResponseParser::toTable(const DatabaseConnector::QueryResult& result) {
    if (result.rows.empty()) {
        return "No results found.\n";
    }
    
    std::ostringstream oss;
    auto widths = calculateColumnWidths(result);
    
    // Верхняя граница
    oss << "+";
    for (auto width : widths) {
        oss << std::string(width + 2, '-') << "+";
    }
    oss << "\n";
    
    // Заголовки
    oss << "|";
    for (size_t i = 0; i < result.columns.size(); ++i) {
        oss << " " << padString(result.columns[i], widths[i]) << " |";
    }
    oss << "\n";
    
    // Разделитель
    oss << "+";
    for (auto width : widths) {
        oss << std::string(width + 2, '=') << "+";
    }
    oss << "\n";
    
    // Данные (ограничение до 15 строк)
    size_t maxRows = std::min(static_cast<size_t>(15), result.rows.size());
    for (size_t rowIndex = 0; rowIndex < maxRows; ++rowIndex) {
        const auto& row = result.rows[rowIndex];
        oss << "|";
        for (size_t i = 0; i < row.size() && i < widths.size(); ++i) {
            oss << " " << padString(row[i], widths[i]) << " |";
        }
        oss << "\n";
    }
    
    // Нижняя граница
    oss << "+";
    for (auto width : widths) {
        oss << std::string(width + 2, '-') << "+";
    }
    oss << std::endl;
    
    // Информация о количестве строк
    if (result.rowCount > 15) {
        oss << "\nShowing first 15 rows of " << result.rowCount << " total rows.\n";
    } else {
        oss << "\nTotal rows: " << result.rowCount << "\n";
    }
    
    return oss.str();
}

static std::string csvEscape(const std::string& field) {
    // RFC 4180: if field contains quotes, commas, or newlines, wrap in quotes
    // and double any existing quotes
    bool needsQuoting = false;
    for (char c : field) {
        if (c == '"' || c == ',' || c == '\n' || c == '\r') {
            needsQuoting = true;
            break;
        }
    }
    if (!needsQuoting) return "\"" + field + "\"";

    std::string escaped = "\"";
    for (char c : field) {
        if (c == '"') escaped += "\"\"";
        else escaped += c;
    }
    escaped += "\"";
    return escaped;
}

std::string ResponseParser::toCSV(const DatabaseConnector::QueryResult& result) {
    std::ostringstream oss;

    // Заголовки
    for (size_t i = 0; i < result.columns.size(); ++i) {
        if (i > 0) oss << ",";
        oss << csvEscape(result.columns[i]);
    }
    oss << "\n";

    // Данные (ограничение до 15 строк)
    size_t maxRows = std::min(static_cast<size_t>(15), result.rows.size());
    for (size_t rowIndex = 0; rowIndex < maxRows; ++rowIndex) {
        const auto& row = result.rows[rowIndex];
        for (size_t i = 0; i < row.size(); ++i) {
            if (i > 0) oss << ",";
            oss << csvEscape(row[i]);
        }
        oss << "\n";
    }

    if (result.rowCount > 15) {
        oss << "\n# Showing first 15 rows of " << result.rowCount << " total rows.\n";
    }

    return oss.str();
}

std::string ResponseParser::toPlainText(const DatabaseConnector::QueryResult& result) {
    std::ostringstream oss;
    
    // Данные (ограничение до 15 строк)
    size_t maxRows = std::min(static_cast<size_t>(15), result.rows.size());
    for (size_t rowIndex = 0; rowIndex < maxRows; ++rowIndex) {
        const auto& row = result.rows[rowIndex];
        for (size_t i = 0; i < result.columns.size() && i < row.size(); ++i) {
            oss << result.columns[i] << ": " << row[i] << "\n";
        }
        oss << "\n";
    }
    
    // Информация о количестве строк
    if (result.rowCount > 15) {
        oss << "Showing first 15 rows of " << result.rowCount << " total rows.\n";
    } else {
        oss << "Total rows: " << result.rowCount << "\n";
    }
    
    return oss.str();
}
