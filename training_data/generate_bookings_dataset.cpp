#include <nlohmann/json.hpp>

#include <algorithm>
#include <fstream>
#include <iostream>
#include <random>
#include <regex>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

using json = nlohmann::json;

struct Column {
    std::string name;
    std::string type;
};

struct Table {
    std::string schema;
    std::string name;
    std::vector<Column> columns;
};

struct ForeignKey {
    std::string src_schema;
    std::string src_table;
    std::vector<std::string> src_cols;

    std::string dst_schema;
    std::string dst_table;
    std::vector<std::string> dst_cols;
};

static std::string trim(const std::string& s) {
    size_t start = 0;
    while (start < s.size() && std::isspace(static_cast<unsigned char>(s[start]))) {
        start++;
    }
    size_t end = s.size();
    while (end > start && std::isspace(static_cast<unsigned char>(s[end - 1]))) {
        end--;
    }
    return s.substr(start, end - start);
}

static bool starts_with(const std::string& s, const std::string& prefix) {
    return s.size() >= prefix.size() && std::equal(prefix.begin(), prefix.end(), s.begin());
}

static std::vector<std::string> split_csv_inside_parens(const std::string& inside) {
    std::vector<std::string> out;
    std::string cur;
    for (char c : inside) {
        if (c == ',') {
            auto t = trim(cur);
            if (!t.empty()) out.push_back(t);
            cur.clear();
        } else {
            cur.push_back(c);
        }
    }
    auto t = trim(cur);
    if (!t.empty()) out.push_back(t);
    return out;
}

static bool is_numeric_type(const std::string& t) {
    std::string s = t;
    std::transform(s.begin(), s.end(), s.begin(), ::tolower);
    return (s.find("int") != std::string::npos) || (s.find("numeric") != std::string::npos) ||
           (s.find("decimal") != std::string::npos) || (s.find("real") != std::string::npos) ||
           (s.find("double") != std::string::npos);
}

static bool is_time_type(const std::string& t) {
    std::string s = t;
    std::transform(s.begin(), s.end(), s.begin(), ::tolower);
    return (s.find("timestamp") != std::string::npos) || (s.find("date") != std::string::npos) ||
           (s.find("time") != std::string::npos);
}

static bool is_bool_type(const std::string& t) {
    std::string s = t;
    std::transform(s.begin(), s.end(), s.begin(), ::tolower);
    return s.find("bool") != std::string::npos;
}

static std::string to_lower_copy(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return std::tolower(c); });
    return s;
}

static std::string table_friendly_name(const Table& t) {
    return to_lower_copy(t.name);
}

static std::string fqtn(const Table& t) {
    return t.schema + "." + t.name;
}

static std::unordered_map<std::string, Table> parse_tables(const std::string& sql_path) {
    std::unordered_map<std::string, Table> tables;

    std::ifstream in(sql_path);
    if (!in.is_open()) {
        throw std::runtime_error("Failed to open SQL dump: " + sql_path);
    }

    std::regex create_re(R"(^CREATE TABLE\s+([a-zA-Z_][\w]*)\.([a-zA-Z_][\w]*)\s*\()",
                         std::regex::icase);

    std::string line;
    bool in_table = false;
    Table cur;

    while (std::getline(in, line)) {
        if (!in_table) {
            std::smatch m;
            if (std::regex_search(line, m, create_re)) {
                std::string schema = m[1].str();
                std::string name = m[2].str();
                if (schema != "bookings") {
                    continue;
                }
                in_table = true;
                cur = Table{schema, name, {}};
            }
            continue;
        }

        std::string t = trim(line);
        if (t == ");" || t == ")") {
            tables[fqtn(cur)] = cur;
            in_table = false;
            continue;
        }

        if (t.empty() || starts_with(t, "--")) {
            continue;
        }

        if (starts_with(to_lower_copy(t), "constraint") || starts_with(to_lower_copy(t), "primary") ||
            starts_with(to_lower_copy(t), "unique") || starts_with(to_lower_copy(t), "check") ||
            starts_with(to_lower_copy(t), "foreign")) {
            continue;
        }

        // Expected: colname type ...
        // Example: book_ref character(6) NOT NULL,
        if (t.back() == ',') t.pop_back();

        std::istringstream iss(t);
        std::string col;
        if (!(iss >> col)) {
            continue;
        }
        std::string type;
        if (!(iss >> type)) {
            continue;
        }

        cur.columns.push_back(Column{col, type});
    }

    return tables;
}

static std::vector<ForeignKey> parse_foreign_keys(const std::string& sql_path) {
    std::vector<ForeignKey> fks;

    std::ifstream in(sql_path);
    if (!in.is_open()) {
        throw std::runtime_error("Failed to open SQL dump: " + sql_path);
    }

    // Pattern like:
    // ADD CONSTRAINT ... FOREIGN KEY (ticket_no, flight_id) REFERENCES bookings.segments(ticket_no, flight_id);
    std::regex fk_re(
        R"(FOREIGN\s+KEY\s*\(([^\)]*)\)\s+REFERENCES\s+([a-zA-Z_][\w]*)\.([a-zA-Z_][\w]*)\s*\(([^\)]*)\))",
        std::regex::icase);

    // Need source table context. In dumps it usually appears after:
    // ALTER TABLE ONLY bookings.<table>
    std::regex alter_re(R"(^ALTER TABLE ONLY\s+([a-zA-Z_][\w]*)\.([a-zA-Z_][\w]*))",
                        std::regex::icase);

    std::string line;
    std::string current_schema;
    std::string current_table;

    while (std::getline(in, line)) {
        std::smatch am;
        if (std::regex_search(line, am, alter_re)) {
            current_schema = am[1].str();
            current_table = am[2].str();
            continue;
        }

        if (current_schema != "bookings") {
            continue;
        }

        std::smatch m;
        if (std::regex_search(line, m, fk_re)) {
            ForeignKey fk;
            fk.src_schema = current_schema;
            fk.src_table = current_table;
            fk.src_cols = split_csv_inside_parens(m[1].str());
            fk.dst_schema = m[2].str();
            fk.dst_table = m[3].str();
            fk.dst_cols = split_csv_inside_parens(m[4].str());
            if (fk.dst_schema == "bookings") {
                fks.push_back(std::move(fk));
            }
        }
    }

    return fks;
}

static int rnd_int(std::mt19937& rng, int lo, int hi) {
    std::uniform_int_distribution<int> dist(lo, hi);
    return dist(rng);
}

static std::string rnd_pick(std::mt19937& rng, const std::vector<std::string>& items) {
    if (items.empty()) return "";
    return items[rnd_int(rng, 0, static_cast<int>(items.size()) - 1)];
}

static void add_example(json& examples, const std::string& nl, const std::string& sql) {
    examples.push_back(json{{"nl", nl}, {"sql", sql}});
}

static void generate_table_examples(json& examples, const Table& t, std::mt19937& rng, int per_table) {
    std::string tf = table_friendly_name(t);
    std::string full = fqtn(t);

    // Basic
    {
        const std::vector<std::string> show_templates = {
            "show all ",
            "list all ",
            "display all ",
            "get all ",
        };
        const std::vector<std::string> count_templates = {
            "count all ",
            "how many ",
            "number of ",
        };
        add_example(examples, rnd_pick(rng, show_templates) + tf, "select * from " + full);
        add_example(examples, rnd_pick(rng, count_templates) + tf, "select count(*) from " + full);
    }

    // Find candidate columns
    std::vector<Column> numeric_cols;
    std::vector<Column> time_cols;
    std::vector<Column> bool_cols;
    std::vector<Column> text_cols;

    for (const auto& c : t.columns) {
        if (is_numeric_type(c.type)) numeric_cols.push_back(c);
        else if (is_time_type(c.type)) time_cols.push_back(c);
        else if (is_bool_type(c.type)) bool_cols.push_back(c);
        else text_cols.push_back(c);
    }

    int generated = 0;
    while (generated < per_table) {
        int kind = rnd_int(rng, 0, 9);
        if (kind == 0 && !numeric_cols.empty()) {
            const auto& c = numeric_cols[rnd_int(rng, 0, static_cast<int>(numeric_cols.size()) - 1)];
            int v = rnd_int(rng, 1, 500);
            add_example(examples,
                        "show " + tf + " where " + c.name + " greater than " + std::to_string(v),
                        "select * from " + full + " where " + c.name + " > " + std::to_string(v));
            generated++;
            continue;
        }
        if (kind == 1 && !numeric_cols.empty()) {
            const auto& c = numeric_cols[rnd_int(rng, 0, static_cast<int>(numeric_cols.size()) - 1)];
            int n = rnd_int(rng, 1, 50);
            add_example(examples,
                        "get " + std::to_string(n) + " " + tf + " ordered by " + c.name,
                        "select * from " + full + " order by " + c.name + " asc limit " +
                            std::to_string(n));
            generated++;
            continue;
        }
        if (kind == 2 && !time_cols.empty()) {
            const auto& c = time_cols[rnd_int(rng, 0, static_cast<int>(time_cols.size()) - 1)];
            add_example(examples, "show " + tf + " ordered by " + c.name,
                        "select * from " + full + " order by " + c.name + " asc");
            generated++;
            continue;
        }
        if (kind == 3 && !bool_cols.empty()) {
            const auto& c = bool_cols[rnd_int(rng, 0, static_cast<int>(bool_cols.size()) - 1)];
            add_example(examples, "show " + tf + " where " + c.name + " is true",
                        "select * from " + full + " where " + c.name + " = true");
            generated++;
            continue;
        }

        if (kind == 4 && !numeric_cols.empty()) {
            const auto& c = numeric_cols[rnd_int(rng, 0, static_cast<int>(numeric_cols.size()) - 1)];
            int lo = rnd_int(rng, 1, 250);
            int hi = rnd_int(rng, lo + 1, lo + 500);
            add_example(examples,
                        "show " + tf + " where " + c.name + " between " + std::to_string(lo) +
                            " and " + std::to_string(hi),
                        "select * from " + full + " where " + c.name + " between " +
                            std::to_string(lo) + " and " + std::to_string(hi));
            generated++;
            continue;
        }

        if (kind == 5 && !text_cols.empty()) {
            const auto& c = text_cols[rnd_int(rng, 0, static_cast<int>(text_cols.size()) - 1)];
            const std::vector<std::string> vals = {"alpha", "beta", "gamma", "delta", "omega"};
            std::string v = rnd_pick(rng, vals);
            add_example(examples,
                        "show " + tf + " where " + c.name + " equals " + v,
                        "select * from " + full + " where " + c.name + " = '" + v + "'");
            generated++;
            continue;
        }

        if (kind == 6 && !time_cols.empty()) {
            const auto& c = time_cols[rnd_int(rng, 0, static_cast<int>(time_cols.size()) - 1)];
            const std::vector<std::string> dates = {"2019-01-01", "2020-01-01", "2021-01-01", "2022-01-01"};
            std::string d1 = rnd_pick(rng, dates);
            std::string d2 = rnd_pick(rng, dates);
            if (d2 < d1) std::swap(d1, d2);
            add_example(examples,
                        "show " + tf + " where " + c.name + " between " + d1 + " and " + d2,
                        "select * from " + full + " where " + c.name + " between '" + d1 +
                            "' and '" + d2 + "'");
            generated++;
            continue;
        }

        if (kind == 7 && !numeric_cols.empty()) {
            const auto& c = numeric_cols[rnd_int(rng, 0, static_cast<int>(numeric_cols.size()) - 1)];
            const std::vector<std::string> agg = {"max", "min", "avg", "sum"};
            std::string fn = rnd_pick(rng, agg);
            add_example(examples,
                        "get " + fn + " of " + c.name + " from " + tf,
                        "select " + fn + "(" + c.name + ") from " + full);
            generated++;
            continue;
        }

        if (kind == 8) {
            int n = rnd_int(rng, 1, 50);
            add_example(examples,
                        "get distinct " + tf + " limited to " + std::to_string(n) + " rows",
                        "select distinct * from " + full + " limit " + std::to_string(n));
            generated++;
            continue;
        }

        // fallback
        int n = rnd_int(rng, 1, 25);
        add_example(examples, "get " + std::to_string(n) + " rows from " + tf,
                    "select * from " + full + " limit " + std::to_string(n));
        generated++;
    }
}

static void generate_join_examples(json& examples,
                                   const std::unordered_map<std::string, Table>& tables,
                                   const std::vector<ForeignKey>& fks,
                                   std::mt19937& rng,
                                   int count) {
    if (fks.empty()) return;

    int produced = 0;
    while (produced < count) {
        const auto& fk = fks[rnd_int(rng, 0, static_cast<int>(fks.size()) - 1)];

        std::string src_key = fk.src_schema + "." + fk.src_table;
        std::string dst_key = fk.dst_schema + "." + fk.dst_table;

        auto src_it = tables.find(src_key);
        auto dst_it = tables.find(dst_key);
        if (src_it == tables.end() || dst_it == tables.end()) {
            produced++;
            continue;
        }

        std::string src_full = src_key;
        std::string dst_full = dst_key;

        // Build join condition
        std::vector<std::string> conds;
        size_t n = std::min(fk.src_cols.size(), fk.dst_cols.size());
        for (size_t i = 0; i < n; i++) {
            conds.push_back("s." + fk.src_cols[i] + " = d." + fk.dst_cols[i]);
        }
        if (conds.empty()) {
            produced++;
            continue;
        }

        std::string on = conds[0];
        for (size_t i = 1; i < conds.size(); i++) {
            on += " and " + conds[i];
        }

        std::string src_nl = to_lower_copy(fk.src_table);
        std::string dst_nl = to_lower_copy(fk.dst_table);
        const std::vector<std::string> join_nl_templates = {
            "join " + src_nl + " with " + dst_nl,
            "show " + src_nl + " with " + dst_nl,
            "list " + src_nl + " and " + dst_nl + " together",
            "get " + src_nl + " including " + dst_nl,
        };
        std::string nl = rnd_pick(rng, join_nl_templates);

        int kind = rnd_int(rng, 0, 4);
        if (kind == 0) {
            std::string sql = "select * from " + src_full + " s join " + dst_full + " d on " + on;
            add_example(examples, nl, sql);
            produced++;
            continue;
        }

        if (kind == 1) {
            int n = rnd_int(rng, 1, 50);
            std::string sql = "select * from " + src_full + " s join " + dst_full +
                              " d on " + on + " limit " + std::to_string(n);
            add_example(examples, nl + " limit " + std::to_string(n), sql);
            produced++;
            continue;
        }

        if (kind == 2) {
            std::string sql = "select count(*) from " + src_full + " s join " + dst_full +
                              " d on " + on;
            add_example(examples, "count rows when joining " + src_nl + " with " + dst_nl, sql);
            produced++;
            continue;
        }

        if (kind == 3) {
            int n = rnd_int(rng, 1, 50);
            std::string sql = "select * from " + src_full + " s join " + dst_full +
                              " d on " + on + " order by 1 asc limit " + std::to_string(n);
            add_example(examples, "get " + std::to_string(n) + " rows from join of " + src_nl +
                                     " and " + dst_nl + " ordered", sql);
            produced++;
            continue;
        }

        // fallback: add a simple filter against join key
        {
            int v = rnd_int(rng, 1, 500);
            std::string sql = "select * from " + src_full + " s join " + dst_full +
                              " d on " + on + " where 1 = 1 and 1 < " + std::to_string(v);
            add_example(examples, "filter joined " + src_nl + " and " + dst_nl, sql);
            produced++;
            continue;
        }
    }
}

int main(int argc, char** argv) {
    std::string sql_path = "data_base/demo-20250901-6m.sql";
    std::string out_path = "training_data/bookings_nl_to_sql_train.json";
    int per_table = 200;
    int join_examples = 300;
    int total_examples_target = -1;
    int seed = 42;

    // CLI:
    // generate_bookings_dataset [--sql path] [--out path] [--examples N] [--per-table N] [--joins N] [--seed N]
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if (a == "--sql" && i + 1 < argc) sql_path = argv[++i];
        else if (a == "--out" && i + 1 < argc) out_path = argv[++i];
        else if (a == "--examples" && i + 1 < argc) total_examples_target = std::stoi(argv[++i]);
        else if (a == "--per-table" && i + 1 < argc) per_table = std::stoi(argv[++i]);
        else if (a == "--joins" && i + 1 < argc) join_examples = std::stoi(argv[++i]);
        else if (a == "--seed" && i + 1 < argc) seed = std::stoi(argv[++i]);
    }

    try {
        auto tables = parse_tables(sql_path);
        auto fks = parse_foreign_keys(sql_path);

        std::mt19937 rng(static_cast<std::mt19937::result_type>(seed));

        if (total_examples_target > 0 && !tables.empty()) {
            int base_examples = static_cast<int>(tables.size()) * 2;
            int remaining = std::max(0, total_examples_target - base_examples);
            int table_examples = static_cast<int>(remaining * 0.92);
            int join_target = remaining - table_examples;

            per_table = (table_examples + static_cast<int>(tables.size()) - 1) / static_cast<int>(tables.size());
            join_examples = join_target;
        }

        json root;
        root["examples"] = json::array();
        auto& examples = root["examples"];

        // Deterministic order
        std::vector<std::string> keys;
        keys.reserve(tables.size());
        for (const auto& kv : tables) keys.push_back(kv.first);
        std::sort(keys.begin(), keys.end());

        for (const auto& k : keys) {
            const auto& t = tables.at(k);
            generate_table_examples(examples, t, rng, per_table);
        }

        generate_join_examples(examples, tables, fks, rng, join_examples);

        std::ofstream out(out_path);
        if (!out.is_open()) {
            std::cerr << "Failed to write output: " << out_path << std::endl;
            return 1;
        }
        out << root.dump(2);

        std::cout << "Wrote dataset: " << out_path << std::endl;
        std::cout << "Tables parsed: " << tables.size() << std::endl;
        std::cout << "Foreign keys parsed: " << fks.size() << std::endl;
        std::cout << "Per-table examples: " << per_table << std::endl;
        std::cout << "Join examples: " << join_examples << std::endl;
        std::cout << "Examples: " << examples.size() << std::endl;

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
