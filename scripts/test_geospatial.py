#!/usr/bin/env python3
"""
Tests du système géospatial
===========================

Ce module fournit des tests pour valider le fonctionnement
des agents géospatiaux IA avec n8n.
"""

import sys
import os
import unittest
from datetime import datetime
from typing import Dict, Any

# Ajouter le chemin des scripts pour l'import
sys.path.append('/opt/geoscripts')

from cadastral_agent import CadastralAgent
from domain_agent import DomainAgent
from urbanism_agent import UrbanismAgent
from environmental_agent import EnvironmentalAgent
from workflow_manager import WorkflowManager


class TestCadastralAgent(unittest.TestCase):
    """Tests pour l'agent cadastral"""
    
    def setUp(self):
        """Initialisation avant chaque test"""
        self.agent = CadastralAgent(target_crs="EPSG:2154")
    
    def test_agent_creation(self):
        """Test de la création de l'agent cadastral"""
        self.assertIsInstance(self.agent, CadastralAgent)
        self.assertEqual(self.agent.target_crs, "EPSG:2154")
    
    def test_agent_methods_exist(self):
        """Test de l'existence des méthodes principales"""
        self.assertTrue(hasattr(self.agent, 'load_parcels'))
        self.assertTrue(hasattr(self.agent, 'validate_parcels'))
        self.assertTrue(hasattr(self.agent, 'detect_cadastral_anomalies'))
        self.assertTrue(hasattr(self.agent, 'predict_parcel_values'))
        self.assertTrue(hasattr(self.agent, 'generate_cadastral_report'))


class TestDomainAgent(unittest.TestCase):
    """Tests pour l'agent domanial"""
    
    def setUp(self):
        """Initialisation avant chaque test"""
        self.agent = DomainAgent()
    
    def test_agent_creation(self):
        """Test de la création de l'agent domanial"""
        self.assertIsInstance(self.agent, DomainAgent)
    
    def test_agent_methods_exist(self):
        """Test de l'existence des méthodes principales"""
        self.assertTrue(hasattr(self.agent, 'load_domain_properties'))
        self.assertTrue(hasattr(self.agent, 'load_concessions'))
        self.assertTrue(hasattr(self.agent, 'identify_strategic_zones'))
        self.assertTrue(hasattr(self.agent, 'analyze_concessions'))
        self.assertTrue(hasattr(self.agent, 'generate_domain_report'))


class TestUrbanismAgent(unittest.TestCase):
    """Tests pour l'agent d'urbanisme"""
    
    def setUp(self):
        """Initialisation avant chaque test"""
        self.agent = UrbanismAgent()
    
    def test_agent_creation(self):
        """Test de la création de l'agent d'urbanisme"""
        self.assertIsInstance(self.agent, UrbanismAgent)
    
    def test_agent_methods_exist(self):
        """Test de l'existence des méthodes principales"""
        self.assertTrue(hasattr(self.agent, 'load_planning_zones'))
        self.assertTrue(hasattr(self.agent, 'load_infrastructure'))
        self.assertTrue(hasattr(self.agent, 'analyze_urban_density'))
        self.assertTrue(hasattr(self.agent, 'identify_development_opportunities'))
        self.assertTrue(hasattr(self.agent, 'generate_urbanism_report'))


class TestEnvironmentalAgent(unittest.TestCase):
    """Tests pour l'agent environnemental"""
    
    def setUp(self):
        """Initialisation avant chaque test"""
        self.agent = EnvironmentalAgent()
    
    def test_agent_creation(self):
        """Test de la création de l'agent environnemental"""
        self.assertIsInstance(self.agent, EnvironmentalAgent)
    
    def test_agent_methods_exist(self):
        """Test de l'existence des méthodes principales"""
        self.assertTrue(hasattr(self.agent, 'load_environmental_data'))
        self.assertTrue(hasattr(self.agent, 'load_monitoring_stations'))
        self.assertTrue(hasattr(self.agent, 'assess_environmental_quality'))
        self.assertTrue(hasattr(self.agent, 'detect_environmental_risks'))
        self.assertTrue(hasattr(self.agent, 'generate_environmental_report'))


class TestWorkflowManager(unittest.TestCase):
    """Tests pour le gestionnaire de workflows"""
    
    def setUp(self):
        """Initialisation avant chaque test"""
        self.manager = WorkflowManager(n8n_url="http://localhost:5678")
    
    def test_manager_creation(self):
        """Test de la création du gestionnaire de workflows"""
        self.assertIsInstance(self.manager, WorkflowManager)
        self.assertEqual(self.manager.n8n_url, "http://localhost:5678")
    
    def test_manager_methods_exist(self):
        """Test de l'existence des méthodes principales"""
        self.assertTrue(hasattr(self.manager, 'create_workflow_instance'))
        self.assertTrue(hasattr(self.manager, 'start_workflow'))
        self.assertTrue(hasattr(self.manager, 'get_workflow_status'))
        self.assertTrue(hasattr(self.manager, 'list_active_workflows'))


def run_all_tests() -> Dict[str, Any]:
    """
    Exécuter tous les tests
    
    Returns:
        Résultats des tests
    """
    print("🧪 Exécution des tests du système géospatial...")
    print("=" * 50)
    
    # Créer un test suite
    test_suite = unittest.TestSuite()
    
    # Ajouter tous les tests
    test_classes = [
        TestCadastralAgent,
        TestDomainAgent,
        TestUrbanismAgent,
        TestEnvironmentalAgent,
        TestWorkflowManager
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Exécuter les tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Résumé des résultats
    test_results = {
        'total_tests': result.testsRun,
        'failures': len(result.failures),
        'errors': len(result.errors),
        'success': result.testsRun - len(result.failures) - len(result.errors),
        'timestamp': datetime.now().isoformat(),
        'status': 'passed' if result.wasSuccessful() else 'failed'
    }
    
    print("\n" + "=" * 50)
    print("RÉSULTATS DES TESTS")
    print("=" * 50)
    print(f"Tests exécutés: {test_results['total_tests']}")
    print(f"Réussis: {test_results['success']}")
    print(f"Échoués: {test_results['failures']}")
    print(f"Erreurs: {test_results['errors']}")
    print(f"Statut: {test_results['status'].upper()}")
    
    return test_results


def run_basic_functionality_tests():
    """
    Exécuter des tests de fonctionnalité basiques
    """
    print("\n🔧 Tests de fonctionnalité basiques...")
    
    try:
        # Test de création des agents
        print("  - Création de l'agent cadastral...", end=" ")
        cadastral_agent = CadastralAgent()
        print("✅")
        
        print("  - Création de l'agent domanial...", end=" ")
        domain_agent = DomainAgent()
        print("✅")
        
        print("  - Création de l'agent urbanisme...", end=" ")
        urbanism_agent = UrbanismAgent()
        print("✅")
        
        print("  - Création de l'agent environnemental...", end=" ")
        environmental_agent = EnvironmentalAgent()
        print("✅")
        
        print("  - Création du gestionnaire de workflows...", end=" ")
        workflow_manager = WorkflowManager(n8n_url="http://localhost:5678")
        print("✅")
        
        print("  - Tous les composants de base sont fonctionnels!")
        return True
        
    except Exception as e:
        print(f"❌ Erreur: {e}")
        return False


def main():
    """
    Fonction principale pour exécuter les tests
    """
    print("Tests du système géospatial IA avec n8n")
    print("=" * 50)
    
    # Exécuter les tests de fonctionnalité basiques
    basic_tests_passed = run_basic_functionality_tests()
    
    if basic_tests_passed:
        # Exécuter les tests unitaires
        test_results = run_all_tests()
        
        print(f"\n🎯 Tests terminés: {test_results['status']}")
        
        if test_results['status'] == 'failed':
            print("⚠️  Certains tests ont échoué, mais le système peut être fonctionnel")
            return 1
        else:
            print("🎉 Tous les tests ont réussi!")
            return 0
    else:
        print("❌ Les tests de base ont échoué")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)