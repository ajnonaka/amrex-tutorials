/*
 * Generic Model Interface with automatic default parameter handling
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <vector>
#include <string>
#include <map>

namespace py = pybind11;

/*
 * FORWARD DECLARATIONS
 * These must match your actual function signatures in main.cpp
 */
#include "bindings.H"

SimulationResult heat_equation_main(int argc, char* argv[]);

/*
 * STANDARD MODEL INTERFACE
 */
class ModelInterface {
public:
    virtual std::vector<std::string> get_pnames() const = 0;
    virtual std::vector<std::string> get_outnames() const = 0;
    virtual std::vector<std::vector<double>> get_param_margpc() const = 0;

    // Run simulation - missing parameters automatically filled with defaults
    virtual std::map<std::string, double> run_single(const std::map<std::string, double>& params = {}) = 0;
};

/*
 * HEAT EQUATION MODEL IMPLEMENTATION
 */
class HeatEquationModel : public ModelInterface {
public:
    std::vector<std::string> get_pnames() const override {
        return {"diffusion", "amplitude", "width"};
    }

    std::vector<std::string> get_outnames() const override {
        return {"max_temperature", "std_temperature"};
    }

    std::vector<std::vector<double>> get_param_margpc() const override {
        return {
            {1.0, 0.2},     // diffusion: [mean, std_dev]
            {1.5, 0.3},     // amplitude: [mean, std_dev]
            {0.01, 0.002}   // width: [mean, std_dev]
        };
    }

    std::map<std::string, double> run_single(const std::map<std::string, double>& params = {}) override {
        auto pnames = get_pnames();
        auto margpc = get_param_margpc();

        /*
         * AUTOMATIC DEFAULT HANDLING
         * Start with defaults (means from marginal PC distributions)
         * Then override with any user-provided parameters
         */
        std::map<std::string, double> full_params;

        // Fill in defaults first (use PC means)
        for (size_t i = 0; i < pnames.size(); ++i) {
            full_params[pnames[i]] = margpc[i][0];  // margpc[i][0] = mean
        }

        // Override with any provided parameters
        for (const auto& [name, value] : params) {
            if (full_params.find(name) != full_params.end()) {
                full_params[name] = value;
            } else {
                throw std::runtime_error("Unknown parameter: " + name);
            }
        }

        /*
         * BUILD COMMAND LINE ARGUMENTS
         * Automatically construct args from parameter map
         */
        std::vector<std::string> args_str = {"./HeatEquation_PythonDriver", "inputs"};

        for (const auto& [name, value] : full_params) {
            args_str.push_back(name + "=" + std::to_string(value));
        }

        /*
         * RUN SIMULATION
         */
        std::vector<char*> args_cstr;
        for (auto& s : args_str) {
            args_cstr.push_back(const_cast<char*>(s.c_str()));
        }
        args_cstr.push_back(nullptr);

        auto result = heat_equation_main(static_cast<int>(args_cstr.size() - 1), args_cstr.data());

        if (!result.success) {
            throw std::runtime_error("Simulation failed");
        }

        /*
         * RETURN RESULTS
         * Map output names to values - fixed initialization syntax
         */
        std::map<std::string, double> output_map;
        output_map["max_temperature"] = result.max_temperature;
        output_map["std_temperature"] = result.std_temperature;

        return output_map;
    }
};

/*
 * GENERIC BATCH RUNNER
 */
py::array_t<double> run_batch_generic(
    ModelInterface& model,
    py::array_t<double> params
) {
    auto pnames = model.get_pnames();
    auto outnames = model.get_outnames();

    auto buf = params.request();
    if (buf.ndim != 2 || buf.shape[1] != static_cast<py::ssize_t>(pnames.size())) {
        throw std::runtime_error("Parameters must be (N, " + std::to_string(pnames.size()) + ") array");
    }

    size_t n_sims = buf.shape[0];
    double *ptr = static_cast<double*>(buf.ptr);

    std::vector<py::ssize_t> shape = {static_cast<py::ssize_t>(n_sims), static_cast<py::ssize_t>(outnames.size())};
    auto result = py::array_t<double>(shape);
    auto result_buf = result.request();
    double *result_ptr = static_cast<double*>(result_buf.ptr);

    for (size_t i = 0; i < n_sims; ++i) {
        // Build parameter map
        std::map<std::string, double> param_map;
        for (size_t j = 0; j < pnames.size(); ++j) {
            param_map[pnames[j]] = ptr[i * pnames.size() + j];
        }

        // Run simulation (defaults automatically handled)
        auto outputs = model.run_single(param_map);

        // Store results
        for (size_t j = 0; j < outnames.size(); ++j) {
            result_ptr[i * outnames.size() + j] = outputs[outnames[j]];
        }
    }

    return result;
}

PYBIND11_MODULE(amrex_heat, m) {
    m.doc() = "Generic AMReX Model Interface with automatic defaults";

    py::class_<ModelInterface>(m, "ModelInterface")
        .def("get_pnames", &ModelInterface::get_pnames)
        .def("get_outnames", &ModelInterface::get_outnames)
        .def("get_param_margpc", &ModelInterface::get_param_margpc)
        .def("run_single", &ModelInterface::run_single,
             "Run single simulation with optional parameters (missing params use defaults)",
             py::arg("params") = py::dict());

    py::class_<HeatEquationModel, ModelInterface>(m, "HeatEquationModel")
        .def(py::init<>());

    m.def("run_batch_generic", &run_batch_generic,
          "Run batch simulations",
          py::arg("model"), py::arg("params"));
}
