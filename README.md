# n8n Geospatial Workflow Runner

This project provides a geospatial workflow runner for n8n using QGIS and Python geospatial libraries.

## Deployment to Render

This project is configured for deployment to Render using the `render.yaml` blueprint.

### Prerequisites

1. A Render account (https://render.com)
2. A GitHub account with this repository forked or cloned

### Deployment Steps

1. Fork this repository to your GitHub account
2. Log in to your Render account
3. Click "New Web Service"
4. Connect your GitHub account and select this repository
5. Render will automatically detect the `render.yaml` file and configure:
   - A web service using the Dockerfile
   - A PostgreSQL database
6. Add the required environment variables:
   - `N8N_ENCRYPTION_KEY` - Generate a secure random string
   - `N8N_USER_MANAGEMENT_JWT_SECRET` - Generate a secure random string
7. Click "Create Web Service"

### Environment Variables

The following environment variables are important for configuration:

- `N8N_RUNNERS_MODE` - Set to "external" to run as a runner
- `N8N_ENCRYPTION_KEY` - Required for encrypting credentials
- `N8N_USER_MANAGEMENT_JWT_SECRET` - Required for user authentication
- Database variables are automatically set from the PostgreSQL instance

### File System

The Docker container has the following directories with write permissions:
- `/files` - General file storage
- `/geodata` - Geospatial data storage
- `/qgis-output` - QGIS processing outputs
- `/tmp/geodata-cache` - Temporary geospatial data cache

### Python Libraries

This runner includes a comprehensive set of geospatial Python libraries:
- GeoPandas for geospatial data manipulation
- Shapely for geometric operations
- Rasterio for raster data processing
- PyProj for coordinate transformations
- GeoAlchemy2 for PostGIS integration
- And many more geospatial libraries

## Local Development

To run locally:

```bash
docker build -t n8n-geospatial .
docker run -p 5678:5678 n8n-geospatial
```

# n8n-render (easy mode)

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy)

## License

This project is licensed under the MIT License.