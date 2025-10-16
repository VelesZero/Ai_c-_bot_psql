#include "src/core/QueryBuilder.h"
#include "src/utils/Logger.h"
#include "src/utils/Utilities.h"
#include <sstream>
#include <algorithm>
#include <regex>

QueryBuilder::QueryBuilder() : limitCount_(-1) {}

QueryBuilder& QueryBuilder::select(const std::vector<std::string>& columns) {
    selectColumns_ = columns;
    return *this;
}

QueryBuilder& QueryBuilder::from(const std::string& table) {
    fromTable_ = sanitize(table);
    return *this;
}

QueryBuilder& QueryBuilder::where(const std::string& condition) {
    whereCondition_ = condition;
    return *this;
}

QueryBuilder& QueryBuilder::orderBy(const std::string& column, bool ascending) {
    orderByClause_ = sanitize(column) + (ascending ? " ASC" : " DESC");
    return *this;
}

QueryBuilder& QueryBuilder::limit(int count) {
    limitCount_ = count;
    return *this;
}

std::string QueryBuilder::build() const {
    std::ostringstream sql;
    
    // SELECT
    sql << "SELECT ";
    if (selectColumns_.empty()) {
        sql << "*";
    } else {
        for (size_t i = 0; i < selectColumns_.size(); ++i) {
            if (i > 0) sql << ", ";
            sql << sanitize(selectColumns_[i]);
        }
    }
    
    // FROM
    if (fromTable_.empty()) {
        Logger::getInstance().error("Table name is required");
        return "";
    }
    sql << " FROM " << fromTable_;
    
    // WHERE
    if (!whereCondition_.empty()) {
        sql << " WHERE " << whereCondition_;
    }
    
    // ORDER BY
    if (!orderByClause_.empty()) {
        sql << " ORDER BY " << orderByClause_;
    }
    
    // LIMIT
    if (limitCount_ > 0) {
        sql << " LIMIT " << limitCount_;
    }
    
    return sql.str();
}

void QueryBuilder::reset() {
    selectColumns_.clear();
    fromTable_.clear();
    whereCondition_.clear();
    orderByClause_.clear();
    limitCount_ = -1;
}

bool QueryBuilder::validateSQL(const std::string& sql) const {
    std::string lower = Utils::toLower(sql);
    
    // Проверка на опасные команды
    std::vector<std::string> dangerousKeywords = {
        "drop", "truncate", "alter", "create", "grant", "revoke"
    };
    
    for (const auto& keyword : dangerousKeywords) {
        if (lower.find(keyword) != std::string::npos) {
            Logger::getInstance().warning("Dangerous keyword detected: " + keyword);
            return false;
        }
    }
    
    // Проверка на SQL injection паттерны
    std::regex injectionPattern(R"((;|\-\-|\/\*|\*\/|xp_|sp_|exec|execute))");
    if (std::regex_search(lower, injectionPattern)) {
        Logger::getInstance().warning("Potential SQL injection detected");
        return false;
    }
    
    return true;
}

std::string QueryBuilder::sanitize(const std::string& input) const {
    std::string result = input;
    
    // Удаление опасных символов
    result.erase(std::remove(result.begin(), result.end(), ';'), result.end());
    result.erase(std::remove(result.begin(), result.end(), '\''), result.end());
    result.erase(std::remove(result.begin(), result.end(), '\"'), result.end());
    
    return Utils::trim(result);
}

bool QueryBuilder::isDangerousKeyword(const std::string& input) const {
    std::string lower = Utils::toLower(input);
    std::vector<std::string> dangerous = {
        "drop", "delete", "truncate", "alter", "create", "insert", "update"
    };
    
    return std::find(dangerous.begin(), dangerous.end(), lower) != dangerous.end();
}