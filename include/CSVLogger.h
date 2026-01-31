#pragma once
#include <fstream>
#include <vector>
#include <string>

class CSVLogger {
public:
    explicit CSVLogger(const std::string& filename): out(filename) {}

    void writeHeader(const std::string& headerLine) {
        out << headerLine << "\n";
    }

    void writeRow(const std::vector<double>& row) {
        for (size_t i = 0; i < row.size(); ++i) {
            out << row[i];
            if (i+1 < row.size()) out << ",";
        }
        out << "\n";
    }

    void flush() {out.flush(); }
private:
    std::ofstream out;
};