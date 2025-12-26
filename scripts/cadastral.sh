#!/bin/bash
# cadastral.sh - Wrapper pour l'agent cadastral
# Usage: ./cadastral.sh <commande> [options]

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_SCRIPT="${SCRIPT_DIR}/cadastral_cli.py"
PYTHON_CMD="${PYTHON_CMD:-python3}"
DATA_DIR="${CADASTRAL_DATA_DIR:-/tmp/cadastral}"

# Créer le répertoire de données si nécessaire
mkdir -p "$DATA_DIR"

# Fonction d'aide
show_help() {
    cat << EOF
Agent Cadastral CLI - Wrapper Bash pour n8n

USAGE:
    ./cadastral.sh <commande> [options]

COMMANDES:
    validate        Valider les parcelles cadastrales
    anomalies       Détecter les anomalies
    consolidate     Consolider les parcelles
    predict-values  Prédire les valeurs foncières
    report          Générer un rapport complet
    changes         Détecter les changements
    stats           Calculer les statistiques

OPTIONS GLOBALES:
    -f, --file      Fichier GeoJSON en entrée
    -o, --output    Fichier de sortie (défaut: stdout)
    -h, --help      Afficher cette aide

EXEMPLES:
    # Valider depuis un fichier
    ./cadastral.sh validate -f parcelles.geojson

    # Valider depuis stdin (pour n8n)
    echo '{"parcels":[...]}' | ./cadastral.sh validate

    # Générer un rapport et sauvegarder
    ./cadastral.sh report -f parcelles.geojson -o rapport.json

    # Pipeline complet
    ./cadastral.sh validate -f data.geojson | jq '.issues[] | select(.issue_count > 0)'

VARIABLES D'ENVIRONNEMENT:
    PYTHON_CMD          Commande Python à utiliser (défaut: python3)
    CADASTRAL_DATA_DIR  Répertoire des données (défaut: /tmp/cadastral)

EOF
}

# Vérifier que Python est disponible
check_python() {
    if ! command -v "$PYTHON_CMD" &> /dev/null; then
        echo "Erreur: Python non trouvé ($PYTHON_CMD)" >&2
        exit 1
    fi
}

# Vérifier que le script Python existe
check_script() {
    if [[ ! -f "$PYTHON_SCRIPT" ]]; then
        echo "Erreur: Script Python non trouvé: $PYTHON_SCRIPT" >&2
        exit 1
    fi
}

# Exécuter une commande
run_command() {
    local cmd="$1"
    shift
    
    check_python
    check_script
    
    "$PYTHON_CMD" "$PYTHON_SCRIPT" "$cmd" "$@"
}

# Parser les arguments
main() {
    if [[ $# -eq 0 ]]; then
        show_help
        exit 0
    fi
    
    local command="$1"
    shift
    
    case "$command" in
        validate|anomalies|consolidate|predict-values|report|changes|stats)
            run_command "$command" "$@"
            ;;
        -h|--help|help)
            show_help
            ;;
        *)
            echo "Erreur: Commande inconnue: $command" >&2
            echo "Utilisez './cadastral.sh --help' pour l'aide" >&2
            exit 1
            ;;
    esac
}

main "$@"