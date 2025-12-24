# n8n Geospatial Workflow Runner - Project Context

## Project Overview

This is an n8n Geospatial Workflow Runner project that provides comprehensive geospatial processing capabilities for the n8n automation platform. It integrates QGIS, Python geospatial libraries, PostGIS, and GRASS GIS to enable sophisticated geospatial workflows in n8n.

The project is designed to run as an external runner for n8n, with specialized Python scripts for comprehensive geospatial analysis, processing, and visualization using QGIS Processing, PostGIS, and GRASS GIS. It's configured for deployment to Render with an accompanying PostgreSQL database with PostGIS extension.

## Architecture & Technologies

- **Base Image**: QGIS official Docker image (qgis/qgis:latest) providing access to QGIS ecosystem
- **n8n Runner**: Version 2.1.1 running in external mode for distributed processing
- **Geospatial Libraries**:
  - QGIS Processing for advanced vector and raster geospatial operations
  - PostGIS for advanced spatial database operations and analysis
  - Python libraries: GeoPandas, Shapely, Rasterio, Xarray, Rioxarray, PyProj, GeoAlchemy2, etc.
  - GRASS GIS for advanced geoprocessing and terrain analysis
  - Scikit-learn for spatial clustering and machine learning
  - Matplotlib, Folium, Plotly for geospatial visualization
- **File Storage**: Dedicated volumes for geospatial data processing and caching

## Key Components

### Dockerfile
- Builds from official QGIS image with full geospatial toolkit
- Installs Node.js 20.x and n8n runner
- Configures comprehensive geospatial Python library stack
- Sets up working directories with appropriate permissions for geospatial operations
- Includes custom geospatial scripts in `/opt/geoscripts`

### Python Scripts in `/scripts`
- `qgis_processing.py`: Comprehensive QGIS Processing wrapper for n8n workflows with advanced algorithms
- `postgis_utils.py`: Advanced PostGIS database operations utilities with spatial analysis functions
- `grass_utils.py`: GRASS GIS processing functions for terrain and advanced geoprocessing
- `health_check.py`: Health check functionality for geospatial services

### Configuration Files
- `render.yaml`: Render deployment configuration with PostgreSQL database and PostGIS extension
- `.env.example`: Example environment variables for deployment
- `requirements.txt`: Comprehensive geospatial Python dependencies with spatial analysis libraries
- `startup.sh`: Container startup script with geospatial environment configuration

## Comprehensive Geospatial Capabilities

### QGIS Processing Functions
- **Vector Operations**: Buffer, clip, overlay (intersection, union, difference), dissolve, spatial join
- **Raster Processing**: Clipping, hillshade, contours, reclassification, resampling
- **Coordinate Reference Systems**: Reprojection, transformation, CRS management
- **Spatial Analysis**: Distance calculations, proximity analysis, geometric operations
- **Terrain Analysis**: DEM processing, watershed analysis, visibility analysis
- **Algorithm Management**: List available algorithms, get algorithm help and parameters
- **Advanced Processing**: Zonal statistics, interpolation, spatial clustering

### PostGIS Functions
- **Database Connection**: Configurable connection management with connection pooling
- **Spatial Queries**: Advanced spatial operators (ST_Intersects, ST_Distance, ST_Buffer, etc.)
- **GeoDataFrame I/O**: Read/write between PostGIS and GeoPandas with optimization
- **Spatial Indexing**: Automatic spatial index creation and management
- **Nearest Neighbor Analysis**: K-nearest neighbor searches with distance calculations
- **Aggregation Functions**: Spatial aggregation, area/length calculations, centroid computation
- **Raster Support**: PostGIS raster operations and analysis
- **Topological Operations**: Topological queries and validation

### GRASS GIS Functions
- **Vector Processing**: Advanced vector operations beyond QGIS capabilities
- **Terrain Analysis**: Slope, aspect, flow accumulation, watershed delineation
- **Raster Analysis**: Advanced raster processing with GRASS capabilities
- **Temporal Processing**: Space-time dataset management

### Geospatial Python Libraries Integration
- **GeoPandas/Geopandas**: Vector data manipulation and analysis
- **Shapely**: Geometric operations and spatial predicates
- **Rasterio/Xarray**: Raster data I/O and analysis
- **PyProj**: Coordinate transformation and CRS operations
- **GeoAlchemy2**: SQLAlchemy integration with PostGIS
- **Scikit-learn**: Spatial clustering (DBSCAN, K-means) and spatial ML
- **GeoPy**: Geocoding and reverse geocoding services
- **Visualization**: Matplotlib, Folium, Plotly for geospatial visualization

## File System Structure for Geospatial Processing
- `/files` - General file storage for inputs and outputs
- `/geodata` - Geospatial data storage with read/write access for processing
- `/qgis-output` - QGIS processing outputs with proper permissions
- `/tmp/geodata-cache` - Temporary geospatial data cache for performance
- `/opt/geoscripts` - Geospatial processing scripts accessible to n8n nodes
- Permissions configured for concurrent geospatial processing operations

## Environment Variables for Geospatial Operations

### Required for Deployment
- `N8N_ENCRYPTION_KEY`: Encryption key for n8n credentials
- `N8N_USER_MANAGEMENT_JWT_SECRET`: JWT secret for authentication
- `N8N_RUNNERS_MODE`: Set to "external" for runner mode

### Database Variables (automatically configured on Render)
- `DB_TYPE`, `DB_POSTGRESDB_HOST`, `DB_POSTGRESDB_PORT`
- `DB_POSTGRESDB_USER`, `DB_POSTGRESDB_PASSWORD`, `DB_POSTGRESDB_DATABASE`

### Geospatial Configuration
- `GDAL_CACHEMAX`, `GDAL_NUM_THREADS`, `PROJ_NETWORK` for performance tuning
- `QT_QPA_PLATFORM=offscreen` for headless QGIS operation
- `XDG_RUNTIME_DIR` for Qt runtime in containerized environment
- Performance settings optimized for geospatial processing workloads

## Building and Running for Geospatial Workflows

### Local Development
```bash
docker build -t n8n-geospatial .
docker run -p 5678:5678 -v ./data:/geodata n8n-geospatial
```

### Render Deployment
- Automatically deployed using `render.yaml` configuration with PostGIS database
- Includes PostgreSQL database setup with PostGIS extension
- Uses Dockerfile for container build with full geospatial stack
- Configures environment variables and health checks for geospatial services

## Development Conventions for Geospatial Workflows

1. **Geospatial Processing**: Python scripts in `/scripts` designed for n8n integration with error handling
2. **QGIS Headless Mode**: All QGIS operations configured for headless execution with proper initialization
3. **Python Environment**: Uses Python 3 with comprehensive geospatial library stack and spatial analysis tools
4. **Security**: Environment variables required for encryption and authentication
5. **File Permissions**: All geospatial data directories configured with write permissions for processing
6. **Error Handling**: Proper exception handling for geospatial operations with meaningful error messages
7. **Performance**: Optimized for geospatial processing with appropriate caching and resource allocation
8. **CRS Management**: Proper handling of coordinate reference systems and reprojections

## Testing and Health Checks

- Health checks available at `/healthz` endpoint for monitoring
- Python-based health check script in `scripts/health_check.py`
- Verifies n8n service and geospatial processing capabilities are responding correctly

## Deployment Notes for Geospatial Applications

- Designed for Render deployment with automatic PostgreSQL database provisioning with PostGIS extension
- Single-click deployment using `render.yaml` configuration with geospatial-optimized settings
- Environment variables configured for different deployment environments with geospatial optimizations
- Automatic scaling considering geospatial processing resource requirements
- Database management includes PostGIS extension and spatial index management