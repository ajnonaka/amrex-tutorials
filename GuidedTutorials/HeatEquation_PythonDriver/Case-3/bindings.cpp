/*
 * Minimal Python bindings for Heat Equation
 *
 * This provides just the essential one-line interface:
 * result = amrex_heat.run_simulation(["./executable", "inputs"])
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

// Forward declaration from main.cpp
struct SimulationResult {
    double max_temperature;
    int final_step;
    double final_time;
    bool success;
};

SimulationResult heat_equation_main(int argc, char* argv[]);

namespace py = pybind11;

PYBIND11_MODULE(amrex_heat, m) {
    m.doc() = "Minimal AMReX Heat Equation Python Interface";

    // Expose SimulationResult struct
    py::class_<SimulationResult>(m, "SimulationResult")
        .def_readonly("max_temperature", &SimulationResult::max_temperature)
        .def_readonly("final_step", &SimulationResult::final_step)
        .def_readonly("final_time", &SimulationResult::final_time)
        .def_readonly("success", &SimulationResult::success);

    // Main simulation function - one-liner interface
    m.def("run_simulation", [](py::list args) {
        // Convert Python list to C++ argc/argv with proper lifetime management
        std::vector<std::string> args_str;
        for (auto item : args) {
            args_str.push_back(py::str(item));
        }

        std::vector<char*> args_cstr;
        for (auto& s : args_str) {
            args_cstr.push_back(const_cast<char*>(s.c_str()));
        }
        args_cstr.push_back(nullptr);  // Null terminate

        return heat_equation_main(static_cast<int>(args_cstr.size() - 1), args_cstr.data());
    }, "Run the heat equation simulation and return results", py::arg("args"));
}