#!/usr/bin/env python3
"""
Système de Gestion des Workflows Géospatiaux IA
================================================

Ce module fournit un système centralisé pour gérer, orchestrer et superviser
les workflows géospatiaux IA dans l'application n8n.
"""

import json
import os
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
from pathlib import Path
import requests
import pandas as pd
import geopandas as gpd

# Configuration de la journalisation
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WorkflowStatus(Enum):
    """Statuts possibles d'un workflow"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class WorkflowType(Enum):
    """Types de workflows géospatiaux"""
    CADASTRAL = "cadastral"
    DOMAINAL = "domanal"
    URBANISM = "urbanism"
    ENVIRONMENTAL = "environmental"
    ANALYTICAL = "analytical"
    REPORTING = "reporting"
    MONITORING = "monitoring"


@dataclass
class WorkflowInstance:
    """Représente une instance de workflow en cours d'exécution"""
    id: str
    name: str
    workflow_type: WorkflowType
    status: WorkflowStatus
    parameters: Dict[str, Any]
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    progress: float = 0.0
    current_step: str = "initial"
    logs: List[str] = field(default_factory=list)


class WorkflowManager:
    """
    Gestionnaire central des workflows géospatiaux IA
    """

    def __init__(self, n8n_url: str = "http://localhost:5678", api_key: Optional[str] = None):
        """
        Initialiser le gestionnaire de workflows

        Args:
            n8n_url: URL de l'instance n8n
            api_key: Clé API pour l'authentification (si nécessaire)
        """
        self.n8n_url = n8n_url
        self.api_key = api_key
        self.headers = {
            "Content-Type": "application/json"
        }
        if api_key:
            self.headers["Authorization"] = f"Bearer {api_key}"

        self.active_workflows: Dict[str, WorkflowInstance] = {}
        self.workflow_templates: Dict[str, Dict[str, Any]] = {}
        self.workflow_history: List[WorkflowInstance] = []

        # Charger les modèles de workflows existants
        self._load_workflow_templates()

    def _load_workflow_templates(self) -> None:
        """Charger les modèles de workflows à partir du répertoire"""
        workflows_dir = Path("workflows")
        if workflows_dir.exists():
            for workflow_file in workflows_dir.glob("*.json"):
                try:
                    with open(workflow_file, 'r', encoding='utf-8') as f:
                        workflow_data = json.load(f)
                        workflow_name = workflow_file.stem
                        self.workflow_templates[workflow_name] = workflow_data
                        logger.info(f"Modèle de workflow chargé: {workflow_name}")
                except Exception as e:
                    logger.error(f"Erreur lors du chargement du workflow {workflow_file}: {e}")

    def create_workflow_instance(
        self,
        workflow_name: str,
        parameters: Dict[str, Any],
        workflow_type: WorkflowType = WorkflowType.ANALYTICAL
    ) -> str:
        """
        Créer une nouvelle instance de workflow

        Args:
            workflow_name: Nom du workflow à exécuter
            parameters: Paramètres du workflow
            workflow_type: Type de workflow

        Returns:
            ID de l'instance créée
        """
        workflow_id = f"wf_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{workflow_name}"
        
        instance = WorkflowInstance(
            id=workflow_id,
            name=workflow_name,
            workflow_type=workflow_type,
            status=WorkflowStatus.PENDING,
            parameters=parameters,
            created_at=datetime.now()
        )

        self.active_workflows[workflow_id] = instance
        logger.info(f"Instance de workflow créée: {workflow_id}")
        
        return workflow_id

    def start_workflow(self, workflow_id: str) -> bool:
        """
        Démarrer l'exécution d'un workflow

        Args:
            workflow_id: ID de l'instance de workflow

        Returns:
            True si le démarrage a réussi, False sinon
        """
        if workflow_id not in self.active_workflows:
            logger.error(f"Workflow non trouvé: {workflow_id}")
            return False

        instance = self.active_workflows[workflow_id]
        instance.status = WorkflowStatus.RUNNING
        instance.started_at = datetime.now()
        instance.current_step = "starting"
        instance.logs.append(f"Démarrage du workflow à {instance.started_at}")

        try:
            # Exécuter le workflow via l'API n8n
            success = self._execute_n8n_workflow(instance)
            if success:
                instance.status = WorkflowStatus.COMPLETED
                instance.completed_at = datetime.now()
                instance.progress = 100.0
                instance.logs.append(f"Workflow terminé avec succès à {instance.completed_at}")
            else:
                instance.status = WorkflowStatus.FAILED
                instance.completed_at = datetime.now()
                instance.logs.append(f"Échec du workflow à {instance.completed_at}")

            # Déplacer de actifs vers historique
            self.workflow_history.append(self.active_workflows.pop(workflow_id))

            return success

        except Exception as e:
            instance.status = WorkflowStatus.FAILED
            instance.completed_at = datetime.now()
            instance.error = str(e)
            instance.logs.append(f"Erreur lors de l'exécution: {e}")
            logger.error(f"Erreur d'exécution du workflow {workflow_id}: {e}")

            # Déplacer de actifs vers historique
            self.workflow_history.append(self.active_workflows.pop(workflow_id))

            return False

    def _execute_n8n_workflow(self, instance: WorkflowInstance) -> bool:
        """
        Exécuter un workflow via l'API n8n

        Args:
            instance: Instance du workflow à exécuter

        Returns:
            True si l'exécution a réussi, False sinon
        """
        try:
            # Trouver le workflow dans les modèles
            if instance.name not in self.workflow_templates:
                raise ValueError(f"Modèle de workflow non trouvé: {instance.name}")

            # Pour l'instant, on simule l'exécution
            # Dans une implémentation réelle, on appellerait l'API n8n
            logger.info(f"Exécution simulée du workflow: {instance.name}")
            
            # Mise à jour du progrès
            instance.progress = 50.0
            instance.current_step = "processing"
            instance.logs.append("Exécution en cours...")

            # Simulation de traitement
            time.sleep(2)  # Simuler un traitement
            
            instance.progress = 100.0
            instance.current_step = "completed"
            instance.result = {"status": "success", "message": "Workflow exécuté avec succès"}
            
            return True

        except Exception as e:
            logger.error(f"Erreur lors de l'exécution n8n: {e}")
            instance.error = str(e)
            return False

    def get_workflow_status(self, workflow_id: str) -> Optional[WorkflowInstance]:
        """
        Obtenir le statut d'une instance de workflow

        Args:
            workflow_id: ID de l'instance de workflow

        Returns:
            Instance du workflow ou None si non trouvée
        """
        if workflow_id in self.active_workflows:
            return self.active_workflows[workflow_id]
        
        # Chercher dans l'historique
        for wf in self.workflow_history:
            if wf.id == workflow_id:
                return wf
        
        return None

    def list_active_workflows(self) -> List[WorkflowInstance]:
        """Lister les workflows actifs"""
        return list(self.active_workflows.values())

    def list_workflow_history(self) -> List[WorkflowInstance]:
        """Lister l'historique des workflows"""
        return self.workflow_history

    def cancel_workflow(self, workflow_id: str) -> bool:
        """
        Annuler un workflow en cours

        Args:
            workflow_id: ID de l'instance de workflow

        Returns:
            True si l'annulation a réussi, False sinon
        """
        if workflow_id not in self.active_workflows:
            return False

        instance = self.active_workflows[workflow_id]
        instance.status = WorkflowStatus.CANCELLED
        instance.completed_at = datetime.now()
        instance.logs.append(f"Workflow annulé à {instance.completed_at}")
        
        # Déplacer vers historique
        self.workflow_history.append(self.active_workflows.pop(workflow_id))
        
        return True

    def execute_workflow_sync(
        self,
        workflow_name: str,
        parameters: Dict[str, Any],
        workflow_type: WorkflowType = WorkflowType.ANALYTICAL
    ) -> Optional[Dict[str, Any]]:
        """
        Exécuter un workflow de manière synchrone

        Args:
            workflow_name: Nom du workflow à exécuter
            parameters: Paramètres du workflow
            workflow_type: Type de workflow

        Returns:
            Résultat du workflow ou None en cas d'erreur
        """
        workflow_id = self.create_workflow_instance(workflow_name, parameters, workflow_type)
        self.start_workflow(workflow_id)
        
        instance = self.get_workflow_status(workflow_id)
        if instance and instance.status == WorkflowStatus.COMPLETED:
            return instance.result
        else:
            return None

    def schedule_workflow(
        self,
        workflow_name: str,
        parameters: Dict[str, Any],
        schedule_time: datetime,
        workflow_type: WorkflowType = WorkflowType.ANALYTICAL
    ) -> str:
        """
        Planifier l'exécution d'un workflow

        Args:
            workflow_name: Nom du workflow à planifier
            parameters: Paramètres du workflow
            schedule_time: Heure d'exécution planifiée
            workflow_type: Type de workflow

        Returns:
            ID de l'instance planifiée
        """
        # Pour l'instant, on crée une instance et on la stocke
        # Dans une implémentation complète, on utiliserait un planificateur
        workflow_id = self.create_workflow_instance(workflow_name, parameters, workflow_type)
        instance = self.active_workflows[workflow_id]
        instance.current_step = f"planned_for_{schedule_time.isoformat()}"
        instance.logs.append(f"Workflow planifié pour {schedule_time}")
        
        logger.info(f"Workflow planifié: {workflow_id} pour {schedule_time}")
        return workflow_id

    def get_workflow_statistics(self) -> Dict[str, Any]:
        """Obtenir les statistiques des workflows"""
        total_completed = len([wf for wf in self.workflow_history if wf.status == WorkflowStatus.COMPLETED])
        total_failed = len([wf for wf in self.workflow_history if wf.status == WorkflowStatus.FAILED])
        total_cancelled = len([wf for wf in self.workflow_history if wf.status == WorkflowStatus.CANCELLED])
        
        return {
            'total_completed': total_completed,
            'total_failed': total_failed,
            'total_cancelled': total_cancelled,
            'total_active': len(self.active_workflows),
            'total_history': len(self.workflow_history),
            'success_rate': total_completed / max(len(self.workflow_history), 1) * 100
        }

    def export_workflow_results(self, workflow_id: str, output_path: str) -> bool:
        """
        Exporter les résultats d'un workflow

        Args:
            workflow_id: ID de l'instance de workflow
            output_path: Chemin de destination pour l'export

        Returns:
            True si l'export a réussi, False sinon
        """
        instance = self.get_workflow_status(workflow_id)
        if not instance or not instance.result:
            return False

        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump({
                    'workflow_id': instance.id,
                    'workflow_name': instance.name,
                    'parameters': instance.parameters,
                    'result': instance.result,
                    'execution_info': {
                        'created_at': instance.created_at.isoformat(),
                        'started_at': instance.started_at.isoformat() if instance.started_at else None,
                        'completed_at': instance.completed_at.isoformat() if instance.completed_at else None,
                        'status': instance.status.value
                    }
                }, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Résultats exportés: {output_path}")
            return True

        except Exception as e:
            logger.error(f"Erreur lors de l'export: {e}")
            return False


class GeospatialWorkflowOrchestrator:
    """
    Orchestrator avancé pour les workflows géospatiaux
    """

    def __init__(self, workflow_manager: WorkflowManager):
        self.workflow_manager = workflow_manager
        self.dependency_graph = {}
        self.workflow_chains = []

    def create_cadastral_analysis_chain(self) -> str:
        """
        Créer une chaîne d'analyse cadastrale complète

        Returns:
            ID de la chaîne de workflows
        """
        chain_id = f"chain_{datetime.now().strftime('%Y%m%d_%H%M%S')}_cadastral"
        
        # Définir la séquence des workflows
        workflow_sequence = [
            {
                'name': 'recherche_parcelle',
                'parameters': {},
                'depends_on': []
            },
            {
                'name': 'validation_parcels',
                'parameters': {},
                'depends_on': ['recherche_parcelle']
            },
            {
                'name': 'prediction_valeur_fonciere',
                'parameters': {},
                'depends_on': ['validation_parcels']
            },
            {
                'name': 'generation_actes',
                'parameters': {},
                'depends_on': ['prediction_valeur_fonciere']
            }
        ]
        
        self.workflow_chains.append({
            'id': chain_id,
            'workflows': workflow_sequence,
            'status': 'defined',
            'created_at': datetime.now()
        })
        
        return chain_id

    def create_environmental_monitoring_chain(self) -> str:
        """
        Créer une chaîne de surveillance environnementale

        Returns:
            ID de la chaîne de workflows
        """
        chain_id = f"chain_{datetime.now().strftime('%Y%m%d_%H%M%S')}_environmental"
        
        # Définir la séquence des workflows
        workflow_sequence = [
            {
                'name': 'surveillance_environnementale',
                'parameters': {},
                'depends_on': []
            },
            {
                'name': 'analyse_pollution_impact',
                'parameters': {},
                'depends_on': ['surveillance_environnementale']
            },
            {
                'name': 'identification_risques',
                'parameters': {},
                'depends_on': ['analyse_pollution_impact']
            },
            {
                'name': 'rapport_environnemental',
                'parameters': {},
                'depends_on': ['identification_risques']
            }
        ]
        
        self.workflow_chains.append({
            'id': chain_id,
            'workflows': workflow_sequence,
            'status': 'defined',
            'created_at': datetime.now()
        })
        
        return chain_id

    def execute_workflow_chain(self, chain_id: str, initial_parameters: Dict[str, Any]) -> bool:
        """
        Exécuter une chaîne de workflows

        Args:
            chain_id: ID de la chaîne de workflows
            initial_parameters: Paramètres initiaux

        Returns:
            True si l'exécution a réussi, False sinon
        """
        chain = next((c for c in self.workflow_chains if c['id'] == chain_id), None)
        if not chain:
            logger.error(f"Chaîne de workflows non trouvée: {chain_id}")
            return False

        chain['status'] = 'running'
        results = {}

        for workflow in chain['workflows']:
            # Vérifier les dépendances
            dependencies_satisfied = True
            for dep in workflow['depends_on']:
                if dep not in results:
                    dependencies_satisfied = False
                    break

            if not dependencies_satisfied:
                logger.warning(f"Dépendances non satisfaites pour {workflow['name']}")
                continue

            # Exécuter le workflow
            logger.info(f"Exécution du workflow: {workflow['name']}")
            
            # Fusionner les paramètres des dépendances avec les paramètres initiaux
            workflow_params = initial_parameters.copy()
            for dep in workflow['depends_on']:
                if dep in results:
                    workflow_params.update(results[dep].get('result', {}))

            workflow_id = self.workflow_manager.create_workflow_instance(
                workflow['name'],
                workflow_params
            )

            success = self.workflow_manager.start_workflow(workflow_id)
            result = self.workflow_manager.get_workflow_status(workflow_id)

            if success and result:
                results[workflow['name']] = result
            else:
                logger.error(f"Échec du workflow {workflow['name']}")
                chain['status'] = 'failed'
                return False

        chain['status'] = 'completed'
        chain['completed_at'] = datetime.now()
        logger.info(f"Chaîne de workflows terminée: {chain_id}")
        return True


def main():
    """
    Fonction principale pour démontrer le système de gestion des workflows
    """
    print("Système de Gestion des Workflows Géospatiaux IA")
    print("=" * 55)

    # Initialiser le gestionnaire de workflows
    workflow_manager = WorkflowManager(n8n_url="http://localhost:5678")
    orchestrator = GeospatialWorkflowOrchestrator(workflow_manager)

    print(f"\n📊 Chargé {len(workflow_manager.workflow_templates)} modèles de workflows")

    # Créer et exécuter un workflow simple
    print("\n🏗️  Création d'un workflow d'analyse cadastrale...")
    workflow_id = workflow_manager.create_workflow_instance(
        "ai_agent_cadastral",
        {
            "parcel_id": "PAR001",
            "analysis_type": "comprehensive"
        },
        WorkflowType.CADASTRAL
    )
    print(f"   → Instance créée: {workflow_id}")

    # Démarrer le workflow
    print(f"\n▶️  Démarrage du workflow...")
    success = workflow_manager.start_workflow(workflow_id)
    print(f"   → Exécution {'réussie' if success else 'échouée'}")

    # Afficher le statut
    status = workflow_manager.get_workflow_status(workflow_id)
    if status:
        print(f"   → Statut: {status.status.value}")
        print(f"   → Progression: {status.progress}%")

    # Créer une chaîne de workflows
    print(f"\n🔗 Création d'une chaîne de workflows cadastraux...")
    chain_id = orchestrator.create_cadastral_analysis_chain()
    print(f"   → Chaîne créée: {chain_id}")

    # Planifier un workflow
    print(f"\n📅 Planification d'un workflow de surveillance...")
    future_time = datetime.now() + timedelta(minutes=1)
    scheduled_id = workflow_manager.schedule_workflow(
        "surveillance_environnementale",
        {"frequency": "hourly", "area": "urban"},
        future_time,
        WorkflowType.MONITORING
    )
    print(f"   → Planifié: {scheduled_id}")

    # Afficher les statistiques
    print(f"\n📈 Statistiques des workflows:")
    stats = workflow_manager.get_workflow_statistics()
    for key, value in stats.items():
        print(f"   → {key}: {value}")

    # Afficher les workflows actifs
    active_workflows = workflow_manager.list_active_workflows()
    print(f"\n🔄 Workflows actifs: {len(active_workflows)}")

    print(f"\n✅ Démonstration du système de gestion des workflows terminée")


if __name__ == "__main__":
    main()