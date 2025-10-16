#ifndef QUERY_BUILDER_H
#define QUERY_BUILDER_H

#include <string>
#include <vector>
#include <map>

class QueryBuilder {
public:
    QueryBuilder();
    
    QueryBuilder& select(const std::vector<std::string>& columns);
    QueryBuilder& from(const std::string& table);
    QueryBuilder& where(const std::string& condition);
    QueryBuilder& orderBy(const std::string& column, bool ascending = true);
    QueryBuilder& limit(int count);
    
    std::string build() const;
    void reset();
    
    // Валидация SQL
    bool validateSQL(const std::string& sql) const;
    std::string sanitize(const std::string& input) const;

private:
    std::vector<std::string> selectColumns_;
    std::string fromTable_;
    std::string whereCondition_;
    std::string orderByClause_;
    int limitCount_;
    
    bool isDangerousKeyword(const std::string& input) const;
};

#endif // QUERY_BUILDER_H