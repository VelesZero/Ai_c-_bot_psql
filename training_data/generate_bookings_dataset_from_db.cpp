#include <nlohmann/json.hpp>
#include <pqxx/pqxx>

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

using json = nlohmann::json;

static std::string to_lower_copy(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return std::tolower(c); });
    return s;
}

// (definition of streaming generator is placed further below, after types and helpers)

static std::string quote_ident(const std::string& ident) {
    std::string out = "\"";
    for (char c : ident) {
        if (c == '"') out += "\"\"";
        else out.push_back(c);
    }
    out += "\"";
    return out;
}

static std::string sql_literal(const std::string& s) {
    std::string out = "'";
    for (char c : s) {
        if (c == '\'') out += "''";
        else out.push_back(c);
    }
    out += "'";
    return out;
}

struct ColumnInfo {
    std::string name;
    std::string data_type;
};

struct TableInfo {
    std::string schema;
    std::string name;
    std::vector<ColumnInfo> columns;
};

static bool is_numeric_type(const std::string& t) {
    const std::string s = to_lower_copy(t);
    return (s.find("int") != std::string::npos) || (s.find("numeric") != std::string::npos) ||
           (s.find("decimal") != std::string::npos) || (s.find("real") != std::string::npos) ||
           (s.find("double") != std::string::npos);
}

static bool is_text_type(const std::string& t) {
    const std::string s = to_lower_copy(t);
    return (s.find("char") != std::string::npos) || (s.find("text") != std::string::npos) ||
           (s.find("json") != std::string::npos) || (s.find("uuid") != std::string::npos);
}

static bool is_time_type(const std::string& t) {
    const std::string s = to_lower_copy(t);
    return (s.find("timestamp") != std::string::npos) || (s.find("date") != std::string::npos) ||
           (s.find("time") != std::string::npos);
}

static std::string build_conninfo(const std::string& host,
                                 int port,
                                 const std::string& dbname,
                                 const std::string& user,
                                 const std::string& password) {
    std::ostringstream oss;
    oss << "host=" << host << " port=" << port << " dbname=" << dbname << " user=" << user;
    if (!password.empty()) {
        oss << " password=" << password;
    }
    oss << " options='-c search_path=bookings,public'";
    return oss.str();
}

static std::vector<TableInfo> load_schema(pqxx::connection& conn, const std::string& schema) {
    std::vector<TableInfo> tables;
    pqxx::work txn(conn);

    pqxx::result t_res = txn.exec_params(
        "SELECT table_schema, table_name "
        "FROM information_schema.tables "
        "WHERE table_type='BASE TABLE' AND table_schema = $1 "
        "ORDER BY table_name",
        schema);

    for (const auto& row : t_res) {
        TableInfo t;
        t.schema = row[0].c_str();
        t.name = row[1].c_str();

        pqxx::result c_res = txn.exec_params(
            "SELECT column_name, data_type "
            "FROM information_schema.columns "
            "WHERE table_schema=$1 AND table_name=$2 "
            "ORDER BY ordinal_position",
            t.schema,
            t.name);

        for (const auto& crow : c_res) {
            ColumnInfo c;
            c.name = crow[0].c_str();
            c.data_type = crow[1].c_str();
            t.columns.push_back(std::move(c));
        }

        tables.push_back(std::move(t));
    }

    txn.commit();
    return tables;
}

static std::vector<std::string> sample_distinct_text(pqxx::connection& conn,
                                                     const TableInfo& t,
                                                     const ColumnInfo& c,
                                                     int limit,
                                                     std::mt19937& rng) {
    std::vector<std::string> vals;
    pqxx::work txn(conn);

    std::string q = "SELECT DISTINCT " + quote_ident(c.name) + " AS v FROM " +
                    quote_ident(t.schema) + "." + quote_ident(t.name) +
                    " WHERE " + quote_ident(c.name) + " IS NOT NULL LIMIT " + std::to_string(limit);

    pqxx::result res = txn.exec(q);
    for (const auto& row : res) {
        if (!row[0].is_null()) {
            vals.push_back(row[0].c_str());
        }
    }

    txn.commit();

    std::shuffle(vals.begin(), vals.end(), rng);
    if (static_cast<int>(vals.size()) > limit) vals.resize(static_cast<size_t>(limit));
    return vals;
}

static std::vector<double> sample_numeric(pqxx::connection& conn,
                                         const TableInfo& t,
                                         const ColumnInfo& c,
                                         int limit,
                                         std::mt19937& rng) {
    std::vector<double> vals;
    pqxx::work txn(conn);

    std::string q = "SELECT " + quote_ident(c.name) + " FROM " + quote_ident(t.schema) + "." +
                    quote_ident(t.name) + " WHERE " + quote_ident(c.name) + " IS NOT NULL LIMIT " +
                    std::to_string(limit);

    pqxx::result res = txn.exec(q);
    for (const auto& row : res) {
        if (!row[0].is_null()) {
            try {
                vals.push_back(std::stod(row[0].c_str()));
            } catch (...) {
            }
        }
    }

    txn.commit();

    std::shuffle(vals.begin(), vals.end(), rng);
    if (static_cast<int>(vals.size()) > limit) vals.resize(static_cast<size_t>(limit));
    return vals;
}

static void add_example(json& examples, const std::string& nl, const std::string& sql) {
    json ex;
    ex["nl"] = nl;
    ex["sql"] = sql;
    examples.push_back(std::move(ex));
}

static std::string rnd_pick(std::mt19937& rng, const std::vector<std::string>& items) {
    if (items.empty()) return "";
    std::uniform_int_distribution<size_t> d(0, items.size() - 1);
    return items[d(rng)];
}

static std::string rnd_int_str(std::mt19937& rng, int lo, int hi) {
    std::uniform_int_distribution<int> d(lo, hi);
    return std::to_string(d(rng));
}

struct JsonStreamWriter {
    std::ofstream out;
    bool first = true;
    size_t count = 0;

    explicit JsonStreamWriter(const std::string& path) : out(path) {}

    bool good() const { return out.is_open() && out.good(); }

    void begin() {
        out << "{\n  \"examples\": [\n";
    }

    void write_example(const std::string& nl, const std::string& sql) {
        json ex;
        ex["nl"] = nl;
        ex["sql"] = sql;
        if (!first) out << ",\n";
        first = false;
        out << "    " << ex.dump();
        count++;
    }

    void end() {
        out << "\n  ]\n}\n";
    }
};

static std::string fqtn(const TableInfo& t);

static void generate_for_table_stream(JsonStreamWriter& writer,
                                      pqxx::connection& conn,
                                      const TableInfo& t,
                                      std::mt19937& rng,
                                      int per_table,
                                      size_t max_total_examples) {
    const std::string table = fqtn(t);

    const std::vector<std::string> v_list = {"list", "show", "display", "print", "get", "fetch"};
    const std::vector<std::string> v_list_ru = {"покажи", "выведи", "показать", "вывести", "дай", "получи", "пожалуйста покажи"};
    const std::vector<std::string> v_all = {"all", "everything"};
    const std::vector<std::string> v_all_ru = {"все", "всё"};
    const std::vector<std::string> v_count = {"count", "number of", "how many", "total"};
    const std::vector<std::string> v_count_ru = {"сколько", "количество", "посчитай", "всего"};
    const std::vector<std::string> v_rows_ru = {"строк", "записей"};
    const std::vector<std::string> v_from = {"from", "in"};
    const std::vector<std::string> v_from_ru = {"из", "в"};

    auto write = [&](const std::string& nl, const std::string& sql) {
        if (writer.count >= max_total_examples) return;
        writer.write_example(nl, sql);
    };

    // Always add a few canonical examples
    write("list all " + t.name, "select * from " + table);
    write("number of " + t.name, "select count(*) from " + table);
    write("get 5 rows from " + t.name, "select * from " + table + " limit 5");

    std::vector<ColumnInfo> text_cols;
    std::vector<ColumnInfo> num_cols;
    std::vector<ColumnInfo> time_cols;
    for (const auto& c : t.columns) {
        if (is_text_type(c.data_type)) text_cols.push_back(c);
        else if (is_numeric_type(c.data_type)) num_cols.push_back(c);
        else if (is_time_type(c.data_type)) time_cols.push_back(c);
    }

    auto any_col = [&]() -> const ColumnInfo* {
        if (t.columns.empty()) return nullptr;
        std::uniform_int_distribution<size_t> idx(0, t.columns.size() - 1);
        return &t.columns[idx(rng)];
    };

    // Cache some samples per column to avoid repeated DB hits
    std::unordered_map<std::string, std::vector<std::string>> text_samples;
    std::unordered_map<std::string, std::vector<double>> num_samples;

    for (size_t i = 0; i < std::min<size_t>(3, text_cols.size()); ++i) {
        text_samples[text_cols[i].name] = sample_distinct_text(conn, t, text_cols[i], 50, rng);
    }
    for (size_t i = 0; i < std::min<size_t>(3, num_cols.size()); ++i) {
        num_samples[num_cols[i].name] = sample_numeric(conn, t, num_cols[i], 50, rng);
    }

    int produced = 3;
    while (produced < per_table && writer.count < max_total_examples) {
        std::uniform_int_distribution<int> intent(0, 10);
        int it = intent(rng);

        if (it == 0) {
            const bool ru = (std::uniform_int_distribution<int>(0, 1)(rng) == 1);
            if (ru) {
                write(rnd_pick(rng, v_list_ru) + " " + rnd_pick(rng, v_all_ru) + " " + t.name,
                      "select * from " + table);
            } else {
                write(rnd_pick(rng, v_list) + " " + rnd_pick(rng, v_all) + " " + t.name,
                      "select * from " + table);
            }
            produced++;
            continue;
        }

        if (it == 1) {
            const bool ru = (std::uniform_int_distribution<int>(0, 1)(rng) == 1);
            if (ru) {
                const std::string phr = rnd_pick(rng, v_count_ru);
                if (phr == "сколько") {
                    write(phr + " " + t.name + " " + rnd_pick(rng, v_rows_ru),
                          "select count(*) from " + table);
                } else {
                    write(phr + " " + t.name,
                          "select count(*) from " + table);
                }
            } else {
                write(rnd_pick(rng, v_count) + " " + t.name,
                      "select count(*) from " + table);
            }
            produced++;
            continue;
        }

        if (it == 2) {
            const std::string n = rnd_int_str(rng, 3, 50);
            const bool ru = (std::uniform_int_distribution<int>(0, 1)(rng) == 1);
            if (ru) {
                write(rnd_pick(rng, v_list_ru) + " " + n + " " + rnd_pick(rng, v_rows_ru) + " " + rnd_pick(rng, v_from_ru) + " " + t.name,
                      "select * from " + table + " limit " + n);
            } else {
                write(rnd_pick(rng, v_list) + " " + n + " rows " + rnd_pick(rng, v_from) + " " + t.name,
                      "select * from " + table + " limit " + n);
            }
            produced++;
            continue;
        }

        if (it == 3) {
            const ColumnInfo* c = any_col();
            if (!c) continue;
            const bool ru = (std::uniform_int_distribution<int>(0, 1)(rng) == 1);
            if (ru) {
                write("выведи " + c->name + " из " + t.name,
                      "select " + c->name + " from " + table);
            } else {
                write("select " + c->name + " from " + t.name,
                      "select " + c->name + " from " + table);
            }
            produced++;
            continue;
        }

        if (it == 4) {
            const ColumnInfo* c = any_col();
            if (!c) continue;
            const bool ru = (std::uniform_int_distribution<int>(0, 1)(rng) == 1);
            if (ru) {
                write("покажи " + t.name + " где " + c->name + " не пусто",
                      "select * from " + table + " where " + c->name + " is not null");
            } else {
                write("show " + t.name + " where " + c->name + " is not null",
                      "select * from " + table + " where " + c->name + " is not null");
            }
            produced++;
            continue;
        }

        if (it == 5 && !text_cols.empty()) {
            const auto& c = text_cols[std::uniform_int_distribution<size_t>(0, text_cols.size() - 1)(rng)];
            auto& vals = text_samples[c.name];
            if (vals.empty()) vals = sample_distinct_text(conn, t, c, 50, rng);
            if (vals.empty()) continue;
            std::string v = rnd_pick(rng, vals);
            std::string v_short = v;
            if (v_short.size() > 24) v_short = v_short.substr(0, 24);
            const bool ru = (std::uniform_int_distribution<int>(0, 1)(rng) == 1);
            if (ru) {
                write("покажи " + t.name + " где " + c.name + " равно " + v_short,
                      "select * from " + table + " where " + c.name + " = " + sql_literal(v));
            } else {
                write("show " + t.name + " where " + c.name + " equals " + v_short,
                      "select * from " + table + " where " + c.name + " = " + sql_literal(v));
            }
            produced++;
            continue;
        }

        if (it == 6 && !text_cols.empty()) {
            const auto& c = text_cols[std::uniform_int_distribution<size_t>(0, text_cols.size() - 1)(rng)];
            auto& vals = text_samples[c.name];
            if (vals.empty()) vals = sample_distinct_text(conn, t, c, 50, rng);
            if (vals.empty()) continue;
            std::string v = rnd_pick(rng, vals);
            if (v.size() > 12) v = v.substr(0, 12);
            const bool ru = (std::uniform_int_distribution<int>(0, 1)(rng) == 1);
            if (ru) {
                write("найди " + t.name + " где " + c.name + " содержит " + v,
                      "select * from " + table + " where " + c.name + " ilike " + sql_literal("%" + v + "%"));
            } else {
                write("find " + t.name + " where " + c.name + " contains " + v,
                      "select * from " + table + " where " + c.name + " ilike " + sql_literal("%" + v + "%"));
            }
            produced++;
            continue;
        }

        if (it == 7 && !num_cols.empty()) {
            const auto& c = num_cols[std::uniform_int_distribution<size_t>(0, num_cols.size() - 1)(rng)];
            auto& vals = num_samples[c.name];
            if (vals.empty()) vals = sample_numeric(conn, t, c, 50, rng);
            if (vals.empty()) continue;
            double v = vals[std::uniform_int_distribution<size_t>(0, vals.size() - 1)(rng)];
            std::ostringstream ss;
            ss << v;
            const bool ru = (std::uniform_int_distribution<int>(0, 1)(rng) == 1);
            if (ru) {
                write("покажи " + t.name + " где " + c.name + " больше " + ss.str(),
                      "select * from " + table + " where " + c.name + " > " + ss.str());
            } else {
                write("show " + t.name + " where " + c.name + " greater than " + ss.str(),
                      "select * from " + table + " where " + c.name + " > " + ss.str());
            }
            produced++;
            continue;
        }

        if (it == 8) {
            const ColumnInfo* c = any_col();
            if (!c) continue;
            const bool desc = (std::uniform_int_distribution<int>(0, 1)(rng) == 1);
            const std::string dir = desc ? "desc" : "asc";
            const bool ru = (std::uniform_int_distribution<int>(0, 1)(rng) == 1);
            if (ru) {
                write("отсортируй " + t.name + " по " + c->name + " " + (desc ? "по убыванию" : "по возрастанию"),
                      "select * from " + table + " order by " + c->name + " " + dir + " limit 50");
            } else {
                write("sort " + t.name + " by " + c->name + " " + dir,
                      "select * from " + table + " order by " + c->name + " " + dir + " limit 50");
            }
            produced++;
            continue;
        }

        if (it == 9 && !num_cols.empty()) {
            const auto& c = num_cols[std::uniform_int_distribution<size_t>(0, num_cols.size() - 1)(rng)];
            const bool ru = (std::uniform_int_distribution<int>(0, 1)(rng) == 1);
            if (ru) {
                write("среднее " + c.name + " в " + t.name,
                      "select avg(" + c.name + ") from " + table);
            } else {
                write("get avg of " + c.name + " from " + t.name,
                      "select avg(" + c.name + ") from " + table);
            }
            produced++;
            continue;
        }

        // group by (simple)
        if (!text_cols.empty()) {
            const auto& c = text_cols[std::uniform_int_distribution<size_t>(0, text_cols.size() - 1)(rng)];
            const bool ru = (std::uniform_int_distribution<int>(0, 1)(rng) == 1);
            if (ru) {
                write("сгруппируй " + t.name + " по " + c.name + " и посчитай",
                      "select " + c.name + ", count(*) from " + table + " group by " + c.name + " order by count desc limit 50");
            } else {
                write("group " + t.name + " by " + c.name + " and count",
                      "select " + c.name + ", count(*) from " + table + " group by " + c.name + " order by count desc limit 50");
            }
            produced++;
        }
    }
}

static std::string fqtn(const TableInfo& t) {
    return t.schema + "." + t.name;
}

static void generate_for_table(json& examples,
                               pqxx::connection& conn,
                               const TableInfo& t,
                               std::mt19937& rng,
                               int per_table) {
    const std::string table = fqtn(t);

    add_example(examples, "list all " + t.name, "select * from " + table);
    add_example(examples, "number of " + t.name, "select count(*) from " + table);
    add_example(examples, "get 5 rows from " + t.name, "select * from " + table + " limit 5");

    std::vector<ColumnInfo> text_cols;
    std::vector<ColumnInfo> num_cols;
    std::vector<ColumnInfo> time_cols;

    for (const auto& c : t.columns) {
        if (is_text_type(c.data_type)) text_cols.push_back(c);
        else if (is_numeric_type(c.data_type)) num_cols.push_back(c);
        else if (is_time_type(c.data_type)) time_cols.push_back(c);
    }

    std::shuffle(text_cols.begin(), text_cols.end(), rng);
    std::shuffle(num_cols.begin(), num_cols.end(), rng);
    std::shuffle(time_cols.begin(), time_cols.end(), rng);

    int produced = 3;

    if (!num_cols.empty()) {
        const auto& c = num_cols.front();
        add_example(examples,
                    "get avg of " + c.name + " from " + t.name,
                    "select avg(" + c.name + ") from " + table);
        produced++;

        auto vals = sample_numeric(conn, t, c, 25, rng);
        if (!vals.empty()) {
            std::uniform_int_distribution<size_t> pick(0, vals.size() - 1);
            double v = vals[pick(rng)];
            std::ostringstream ss;
            ss << v;
            add_example(examples,
                        "show " + t.name + " where " + c.name + " greater than " + ss.str(),
                        "select * from " + table + " where " + c.name + " > " + ss.str());
            produced++;
        }
    }

    if (!text_cols.empty()) {
        const auto& c = text_cols.front();
        auto vals = sample_distinct_text(conn, t, c, 25, rng);
        if (!vals.empty()) {
            std::uniform_int_distribution<size_t> pick(0, vals.size() - 1);
            std::string v = vals[pick(rng)];
            std::string v_short = v;
            if (v_short.size() > 24) v_short = v_short.substr(0, 24);
            add_example(examples,
                        "show " + t.name + " where " + c.name + " equals " + v_short,
                        "select * from " + table + " where " + c.name + " = " + sql_literal(v));
            produced++;
        }
    }

    while (produced < per_table) {
        std::uniform_int_distribution<int> kind(0, 3);
        int k = kind(rng);
        if (k == 0 && t.columns.size() >= 1) {
            std::uniform_int_distribution<size_t> idx(0, t.columns.size() - 1);
            const auto& c = t.columns[idx(rng)];
            add_example(examples,
                        "select " + c.name + " from " + t.name,
                        "select " + c.name + " from " + table);
            produced++;
        } else if (k == 1 && t.columns.size() >= 1) {
            std::uniform_int_distribution<size_t> idx(0, t.columns.size() - 1);
            const auto& c = t.columns[idx(rng)];
            add_example(examples,
                        "count where " + c.name + " is not null in " + t.name,
                        "select count(*) from " + table + " where " + c.name + " is not null");
            produced++;
        } else if (k == 2) {
            int lim = 10;
            add_example(examples,
                        "get " + std::to_string(lim) + " rows from " + t.name,
                        "select * from " + table + " limit " + std::to_string(lim));
            produced++;
        } else {
            add_example(examples,
                        "list all " + t.name,
                        "select * from " + table);
            produced++;
        }
    }
}

int main(int argc, char** argv) {
    std::string host = "localhost";
    int port = 5432;
    std::string dbname = "ai_db";
    std::string user = "ai_user";
    std::string password = "123";
    std::string schema = "bookings";
    std::string out_path = "training_data/bookings_nl_to_sql_from_db.json";
    int examples_target = 100000;
    int seed = 42;

    // CLI:
    // generate_bookings_dataset_from_db [--host H] [--port P] [--db NAME] [--user U] [--password PW]
    //                                 [--schema S] [--out PATH] [--examples N] [--seed N]
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--host" && i + 1 < argc) host = argv[++i];
        else if (a == "--port" && i + 1 < argc) port = std::stoi(argv[++i]);
        else if (a == "--db" && i + 1 < argc) dbname = argv[++i];
        else if (a == "--user" && i + 1 < argc) user = argv[++i];
        else if (a == "--password" && i + 1 < argc) password = argv[++i];
        else if (a == "--schema" && i + 1 < argc) schema = argv[++i];
        else if (a == "--out" && i + 1 < argc) out_path = argv[++i];
        else if (a == "--examples" && i + 1 < argc) examples_target = std::stoi(argv[++i]);
        else if (a == "--seed" && i + 1 < argc) seed = std::stoi(argv[++i]);
    }

    try {
        std::mt19937 rng(static_cast<std::mt19937::result_type>(seed));

        const std::string conninfo = build_conninfo(host, port, dbname, user, password);
        pqxx::connection conn(conninfo);

        if (!conn.is_open()) {
            std::cerr << "Failed to connect to database" << std::endl;
            return 1;
        }

        auto tables = load_schema(conn, schema);
        if (tables.empty()) {
            std::cerr << "No tables found in schema: " << schema << std::endl;
            return 1;
        }

        const size_t max_total_examples = static_cast<size_t>(std::max(1, examples_target));
        int per_table = std::max(50, static_cast<int>(max_total_examples / static_cast<size_t>(tables.size())));

        JsonStreamWriter writer(out_path);
        if (!writer.good()) {
            std::cerr << "Failed to write output: " << out_path << std::endl;
            return 1;
        }

        writer.begin();

        for (const auto& t : tables) {
            generate_for_table_stream(writer, conn, t, rng, per_table, max_total_examples);
            if (writer.count >= max_total_examples) break;
        }

        writer.end();

        std::cout << "Wrote dataset: " << out_path << std::endl;
        std::cout << "Schema: " << schema << std::endl;
        std::cout << "Tables: " << tables.size() << std::endl;
        std::cout << "Examples: " << writer.count << std::endl;
        std::cout << "Per-table examples: " << per_table << std::endl;

        std::_Exit(0);
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
