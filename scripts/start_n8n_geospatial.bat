@echo off
setlocal

:: Define constants
set "COMPOSE_FILE=docker-compose.yml"
set "DOCKER_PROJECT_NAME=n8n-geospatial"

:: Function to display usage
:usage
echo Usage: %~nx0 [start^|stop^|restart^|logs^|build^|up^|down]
goto :eof

:: Check for command argument
if "%~1"=="" (
    call :usage
    exit /b 1
)

:: Execute command based on argument
if /i "%~1"=="start" (
    docker-compose -f %COMPOSE_FILE% -p %DOCKER_PROJECT_NAME% up -d --build --remove-orphans
) else if /i "%~1"=="stop" (
    docker-compose -f %COMPOSE_FILE% -p %DOCKER_PROJECT_NAME% stop
) else if /i "%~1"=="restart" (
    docker-compose -f %COMPOSE_FILE% -p %DOCKER_PROJECT_NAME% restart
) else if /i "%~1"=="logs" (
    docker-compose -f %COMPOSE_FILE% -p %DOCKER_PROJECT_NAME% logs -f
) else if /i "%~1"=="build" (
    docker-compose -f %COMPOSE_FILE% -p %DOCKER_PROJECT_NAME% build
) else if /i "%~1"=="up" (
    docker-compose -f %COMPOSE_FILE% -p %DOCKER_PROJECT_NAME% up -d
) else if /i "%~1"=="down" (
    docker-compose -f %COMPOSE_FILE% -p %DOCKER_PROJECT_NAME% down
) else (
    echo Unknown command: %~1
    call :usage
    exit /b 1
)

endlocal
exit /b 0