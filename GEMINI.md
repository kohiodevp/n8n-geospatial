# n8n Geospatial Workflow Runner

## Project Overview

This project provides a comprehensive and powerful geospatial workflow runner for n8n. It leverages the capabilities of QGIS, PostGIS, and a rich ecosystem of Python geospatial libraries to automate complex data processing and analysis tasks. The entire system is containerized using Docker, making it portable, scalable, and easy to deploy.

The core of the project is a customized n8n instance running in a Docker container. This instance is augmented with a suite of Python scripts and a dedicated PostGIS database, enabling it to perform advanced geospatial operations that are not available in the standard n8n distribution.

The project is designed to be highly modular and extensible. It includes a set of pre-built "Geospatial Agents" for common use cases such as cadastral analysis, urban planning, and environmental monitoring. These agents are implemented as a combination of n8n workflows and Python scripts, and they can be easily customized or extended to meet specific project requirements.

## Building and Running

The project is managed using Docker Compose, which simplifies the process of building and running the application. The main commands are provided as shell scripts for both Windows and Linux/macOS environments.

**Key commands:**

*   **Start the system:**
    *   Windows: `.\scripts\start_n8n_geospatial.bat start`
    *   Linux/macOS: `./scripts/start_n8n_geospatial.sh start`
*   **Stop the system:**
    *   Windows: `.\scripts\start_n8n_geospatial.bat stop`
    *   Linux/macOS: `./scripts/start_n8n_geospatial.sh stop`
*   **View logs:**
    *   Windows: `.\scripts\start_n8n_geospatial.bat logs`
    *   Linux/macOS: `./scripts/start_n8n_geospatial.sh logs`
*   **Run tests:**
    *   Windows: `.\scripts\test_geospatial_agents.bat run-all`
    *   Linux/macOS: `./scripts/test_geospatial_agents.sh run-all`

The application will be available at `http://localhost:5678`.

## Development Conventions

The project follows a set of conventions to ensure code quality, consistency, and maintainability.

*   **Configuration:** All configuration is managed through environment variables defined in the `.env` file. This allows for easy customization of the application without modifying the source code.
*   **Workflows:** n8n workflows are stored as JSON files in the `workflows` directory. These workflows are automatically imported into n8n when the application starts.
*   **Scripts:** Custom Python scripts are located in the `scripts` directory. These scripts are used to perform tasks that are not easily accomplished using n8n's built-in nodes.
*   **Database:** The project uses a PostGIS database for storing and querying geospatial data. Database initialization scripts are located in the `init-scripts` directory.
*   **Testing:** The project includes a set of tests for the Geospatial Agents. These tests are located in the `scripts` directory and can be run using the `test_geospatial_agents.bat` or `test_geospatial_agents.sh` scripts.
