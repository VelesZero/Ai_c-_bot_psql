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
    
    // Данные
    for (const auto& row : result.rows) {
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
    oss << "\n";
    
    oss << "\nTotal rows: " << result.rowCount << "\n";
    
    return oss.str();
}

std::string ResponseParser::toCSV(const DatabaseConnector::QueryResult& result) {
    std::ostringstream oss;
    
    // Заголовки
    for (size_t i = 0; i < result.columns.size(); ++i) {
        if (i > 0) oss << ",";
        oss << "\"" << result.columns[i] << "\"";
    }
    oss << "\n";
    
    // Данные
    for (const auto& row : result.rows) {
        for (size_t i = 0; i < row.size(); ++i) {
            if (i > 0) oss << ",";
            oss << "\"" << row[i] << "\"";
        }
        oss << "\n";
    }
    
    return oss.str();
}

std::string ResponseParser::toPlainText(const DatabaseConnector::QueryResult& result) {
    std::ostringstream oss;
    
    for (const auto& row : result.rows) {
        for (size_t i = 0; i < result.columns.size() && i < row.size(); ++i) {
            oss << result.columns[i] << ": " << row[i] << "\n";
        }
        oss << "\n";
    }
    
    oss << "Total rows: " << result.rowCount << "\n";
    
    return oss.str();
}