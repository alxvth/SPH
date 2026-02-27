// Author: Alexander Vieth

#include "EvaluationSettings.hpp"
#include "RunEvaluation.hpp"

#include <cstdlib>
#include <iostream>
#include <string>

#include <fmt/ostream.h>

int main(int argc, char* argv[]) {
   
    if (argc != 2) {
        fmt::println(std::cerr, "Usage: SPHEvaluation <path_to_json_file>. You did not provide a settings file.");
        return EXIT_FAILURE;
    }

    const std::string settingsFilePath = argv[1];
    auto [settings, settingsSuccess] = sph::eval::readSettingsFromFile(settingsFilePath);

    if (!settingsSuccess) {
        fmt::println(std::cerr, "Settings file at {0} not found.", settingsFilePath);
        return EXIT_FAILURE;
    }

    sph::eval::runEvaluation(settings);
    
    return EXIT_SUCCESS;
}
