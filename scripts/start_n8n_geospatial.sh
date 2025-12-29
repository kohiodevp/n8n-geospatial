#!/bin/bash
# Script de démarrage avancé pour n8n Geospatial Workflow Runner
# Ce script démarre tous les services nécessaires dans le bon ordre

set -e  # Arrêter en cas d'erreur

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
DOCKER_COMPOSE_FILE="$PROJECT_ROOT/docker-compose.yml"
ENV_FILE="$PROJECT_ROOT/.env"

# Fonction d'aide
show_help() {
    cat << EOF
Script de démarrage pour n8n Geospatial Workflow Runner

USAGE:
    ./start_n8n_geospatial.sh [commande] [options]

COMMANDES:
    start           Démarrer tous les services (défaut)
    stop            Arrêter tous les services
    restart         Redémarrer tous les services
    status          Vérifier l'état des services
    logs            Afficher les logs des services
    logs-follow     Afficher les logs en continu
    build           Reconstruire les images Docker
    reset           Arrêter et supprimer tous les conteneurs et volumes
    health-check    Vérifier la santé de l'application

OPTIONS:
    -e, --env       Chemin vers le fichier .env (défaut: ../.env)
    -f, --file      Chemin vers le fichier docker-compose.yml
    -h, --help      Afficher cette aide

EXEMPLES:
    # Démarrer tous les services
    ./start_n8n_geospatial.sh start

    # Démarrer en mode détaché
    ./start_n8n_geospatial.sh start -d

    # Afficher les logs en continu
    ./start_n8n_geospatial.sh logs-follow

    # Redémarrer les services
    ./start_n8n_geospatial.sh restart

VARIABLES D'ENVIRONNEMENT:
    N8N_HOST        Hôte pour n8n (défaut: 0.0.0.0)
    N8N_PORT        Port pour n8n (défaut: 5678)
    POSTGRES_DB     Nom de la base de données (défaut: n8n)
    POSTGRES_USER   Utilisateur de la base (défaut: n8n)
    POSTGRES_PASSWORD Mot de passe de la base (défaut: n8npassword)

EOF
}

# Fonction pour vérifier les dépendances
check_dependencies() {
    local missing_deps=()

    if ! command -v docker &> /dev/null; then
        missing_deps+=("docker")
    fi

    if ! command -v docker-compose &> /dev/null; then
        if ! command -v docker &> /dev/null || ! docker compose version &> /dev/null; then
            missing_deps+=("docker-compose")
        fi
    fi

    if [[ ${#missing_deps[@]} -gt 0 ]]; then
        echo "Erreur: Les dépendances suivantes sont manquantes: ${missing_deps[*]}" >&2
        exit 1
    fi
}

# Fonction pour charger le fichier .env
load_env_file() {
    if [[ -f "$ENV_FILE" ]]; then
        echo "Chargement du fichier .env: $ENV_FILE"
        set -a
        source "$ENV_FILE"
        set +a
    else
        echo "⚠️  Fichier .env non trouvé: $ENV_FILE" >&2
        echo "   Utilisation des valeurs par défaut..." >&2
    fi
}

# Fonction pour attendre que PostGIS soit prêt
wait_for_postgis() {
    echo "Attente du service PostGIS..."
    local max_attempts=30
    local attempt=1
    
    until docker-compose -f "$DOCKER_COMPOSE_FILE" exec postgis pg_isready > /dev/null 2>&1
    do
        if [[ $attempt -ge $max_attempts ]]; then
            echo "Erreur: PostGIS n'a pas démarré après $max_attempts tentatives" >&2
            exit 1
        fi
        
        echo "Attente PostGIS... ($attempt/$max_attempts)"
        sleep 5
        ((attempt++))
    done
    
    echo "✅ PostGIS est prêt"
}

# Fonction pour attendre que Redis soit prêt
wait_for_redis() {
    echo "Attente du service Redis..."
    local max_attempts=20
    local attempt=1
    
    until docker-compose -f "$DOCKER_COMPOSE_FILE" exec redis redis-cli ping > /dev/null 2>&1
    do
        if [[ $attempt -ge $max_attempts ]]; then
            echo "Erreur: Redis n'a pas démarré après $max_attempts tentatives" >&2
            exit 1
        fi
        
        echo "Attente Redis... ($attempt/$max_attempts)"
        sleep 3
        ((attempt++))
    done
    
    echo "✅ Redis est prêt"
}

# Fonction pour vérifier la santé de n8n
check_n8n_health() {
    echo "Vérification de la santé de n8n..."
    local max_attempts=30
    local attempt=1
    local n8n_url="http://localhost:5678"
    
    while [[ $attempt -le $max_attempts ]]; do
        if curl -f -s "$n8n_url/healthz" > /dev/null 2>&1; then
            echo "✅ n8n est prêt et sain"
            return 0
        fi
        
        echo "Attente de n8n... ($attempt/$max_attempts)"
        sleep 10
        ((attempt++))
    done
    
    echo "⚠️  n8n n'a pas démarré correctement après $max_attempts tentatives" >&2
    return 1
}

# Commande start
cmd_start() {
    echo " démarrage de n8n Geospatial Workflow Runner "
    echo "==============================================="
    
    check_dependencies
    load_env_file
    
    echo "Construction des images Docker..."
    docker-compose -f "$DOCKER_COMPOSE_FILE" build
    
    echo "Démarrage des services..."
    docker-compose -f "$DOCKER_COMPOSE_FILE" up -d
    
    # Attendre que les services soient prêts
    wait_for_postgis
    wait_for_redis
    check_n8n_health
    
    echo ""
    echo "✅ n8n Geospatial Workflow Runner est démarré avec succès !"
    echo "accès à l'interface: http://localhost:5678"
    echo "Utilisateur: ${N8N_BASIC_AUTH_USER:-admin}"
    echo "Mot de passe: ${N8N_BASIC_AUTH_PASSWORD:-cadastre2024}"
}

# Commande stop
cmd_stop() {
    echo "Arrêt de n8n Geospatial Workflow Runner..."
    docker-compose -f "$DOCKER_COMPOSE_FILE" down
    echo "✅ Services arrêtés"
}

# Commande restart
cmd_restart() {
    cmd_stop
    sleep 5
    cmd_start
}

# Commande status
cmd_status() {
    echo "État des services:"
    docker-compose -f "$DOCKER_COMPOSE_FILE" ps
}

# Commande logs
cmd_logs() {
    docker-compose -f "$DOCKER_COMPOSE_FILE" logs
}

# Commande logs-follow
cmd_logs_follow() {
    docker-compose -f "$DOCKER_COMPOSE_FILE" logs -f
}

# Commande build
cmd_build() {
    echo "Reconstruction des images Docker..."
    docker-compose -f "$DOCKER_COMPOSE_FILE" build --no-cache
    echo "✅ Images reconstruites"
}

# Commande reset
cmd_reset() {
    echo "Arrêt et suppression de tous les conteneurs et volumes..."
    docker-compose -f "$DOCKER_COMPOSE_FILE" down -v
    echo "✅ Conteneurs et volumes supprimés"
}

# Commande health-check
cmd_health_check() {
    echo "Vérification de la santé de l'application..."
    
    # Vérifier l'état des services
    cmd_status
    
    # Vérifier la santé de n8n
    if check_n8n_health; then
        echo "✅ Application saine"
        return 0
    else
        echo "❌ Application non saine"
        return 1
    fi
}

# Parser les arguments
main() {
    local command="${1:-start}"
    
    case "$command" in
        start)
            cmd_start
            ;;
        stop)
            cmd_stop
            ;;
        restart)
            cmd_restart
            ;;
        status)
            cmd_status
            ;;
        logs)
            cmd_logs
            ;;
        logs-follow)
            cmd_logs_follow
            ;;
        build)
            cmd_build
            ;;
        reset)
            cmd_reset
            ;;
        health-check)
            cmd_health_check
            ;;
        -h|--help|help)
            show_help
            ;;
        *)
            echo "Erreur: Commande inconnue: $command" >&2
            echo "Utilisez './start_n8n_geospatial.sh --help' pour l'aide" >&2
            exit 1
            ;;
    esac
}

main "$@"