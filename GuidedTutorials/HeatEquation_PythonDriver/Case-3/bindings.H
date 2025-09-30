#ifndef SIMULATION_RESULT_H
#define SIMULATION_RESULT_H

struct SimulationResult {
    double max_temperature;
    double std_temperature;  // Add this field if you want it
    int final_step;
    double final_time;
    bool success;
};

// Function declaration
SimulationResult heat_equation_main(int argc, char* argv[]);

#endif // SIMULATION_RESULT_H
