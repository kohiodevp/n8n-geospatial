# n8n Geospatial Workflow Runner - Project Context

## Project Overview

This is a sophisticated geospatial workflow automation system built on top of n8n, featuring AI-powered geospatial agents for cadastral, domain, urban planning, environmental, and transportation analysis. The project combines n8n's workflow automation capabilities with Python geospatial libraries (QGIS, GeoPandas, Shapely, Rasterio, etc.) to enable complex geospatial processing workflows.

### Key Features
- **AI Geospatial Agents**: Cadastral, domain, urbanism, environmental, and transportation agents
- **Advanced Spatial Analysis**: Geometric validation, anomaly detection, spatial clustering
- **PostGIS Integration**: Spatial database for storage and analysis
- **QGIS Processing**: Advanced spatial analysis tools
- **Machine Learning**: Classification, prediction, and anomaly detection algorithms
- **Automated Workflow Deployment**: All 22 geospatial workflows automatically loaded
- **Complete Automation**: n8n workflows for complex processes

### Architecture
The system uses a Docker-based architecture with:
- **n8n-geospatial**: Main service with geospatial libraries
- **postgis**: PostgreSQL database with PostGIS extension
- **redis**: Queue management for workflows
- **nginx**: Reverse proxy (optional for production)

## Building and Running

### Prerequisites
- Docker and Docker Compose
- Windows (batch scripts provided) or Linux/macOS
- 4GB RAM minimum (8GB recommended for intensive processing)
- 10GB disk space available

### Quick Start
```bash
# On Windows
.\scripts\start_n8n_geospatial.bat start

# On Linux/macOS
./scripts/start_n8n_geospatial.sh start
```

### Access the Interface
- URL: http://localhost:5678
- Username: admin
- Password: cadastre2024

### Development Mode
```bash
# Start in development mode
.\scripts\dev_n8n_geospatial.bat dev-start

# Reload workflows
.\scripts\dev_n8n_geospatial.bat dev-reload

# Access container shell
.\scripts\dev_n8n_geospatial.bat dev-shell
```

### Testing
```bash
# Run geospatial agent tests
.\scripts\test_geospatial_agents.bat run-all
```

## Development Conventions

### Project Structure
- `/scripts/` - Contains Python agent implementations and utility scripts
- `/data/` - File storage for workflow data
- `/geodata/` - Geospatial data files
- `/workflows/` - n8n workflow definitions
- `/init-scripts/` - Database initialization scripts
- `/docs/` - Documentation files

### Agent Structure
Each geospatial agent follows a consistent pattern:
- Python implementation with dataclasses and enums
- Validation rules and spatial analysis capabilities
- Integration with geospatial libraries (GeoPandas, Shapely, etc.)
- Caching mechanisms for performance optimization

### Environment Configuration
The system uses a comprehensive `.env` file with:
- Authentication and security settings
- Database configuration (PostGIS)
- Runner configuration for external processing
- Geospatial-specific settings (GDAL, PROJ, QGIS)
- Performance optimization parameters

### Key Scripts
- `start_n8n_geospatial.bat` - Basic system management
- `dev_n8n_geospatial.bat` - Development features
- `test_geospatial_agents.bat` - Agent testing
- `optimize_project.bat` - Optimization tools
- `system_check.bat` - System status verification

## Technology Stack

### Backend
- **n8n**: Workflow automation platform
- **QGIS**: Geospatial processing engine
- **PostGIS**: Spatial database
- **Redis**: Queue management

### Python Libraries
- **GeoPandas**: Geospatial data manipulation
- **Shapely**: Geometric operations
- **Rasterio**: Raster data processing
- **PyProj**: Coordinate reference system transformations
- **Scikit-learn**: Machine learning algorithms
- **SciPy/Numpy**: Scientific computing

### Infrastructure
- **Docker**: Containerization
- **Docker Compose**: Multi-container orchestration
- **Nginx**: Reverse proxy (production)

## Key Components

### Cadastral Agent
- Geometric validation of parcels
- Cadastral anomaly detection
- Parcel consolidation
- Property value prediction
- Neighborhood analysis

### Domain Agent
- Domain property management
- Concession analysis
- Strategic zone identification
- Management optimization

### Urbanism Agent
- Urban density analysis
- Development opportunity identification
- Infrastructure capacity assessment
- Urban growth prediction
- Accessibility analysis
- Development scenario simulation

### Environmental Agent
- Environmental quality assessment
- Risk zone detection
- Biodiversity hotspot analysis
- Environmental trend prediction
- Ecosystem services evaluation
- Conservation priority identification
- Pollution impact analysis

### Workflow Management System
- Advanced orchestration capabilities
- Workflow chaining and scheduling
- Execution monitoring and statistics
- Integration with n8n API

## Security Considerations
- Basic authentication enabled by default
- Encryption key for sensitive data
- Node.js function allow-list for security
- Secure JWT secret for authentication
- Disabled potentially dangerous features

## Deployment
The project includes deployment configurations for Render.com with:
- Automated database provisioning
- Environment variable management
- Health check configuration
- Resource allocation settings

## Workflows
The system includes 22 pre-built geospatial workflows:
- AI Agent Cadastral
- AI Agent Domanial
- AI Agent Urbanism
- Environmental Surveillance
- Parcel Consolidation
- Urban Planning Analysis
- Property Value Prediction
- And many more...

All workflows are automatically loaded by n8n from the `/home/node/.n8n/workflows/` directory.