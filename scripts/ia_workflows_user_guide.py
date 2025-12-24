#!/usr/bin/env python3
"""
Guide Utilisateur pour les Workflows IA Géospatiaux
===================================================

Ce module fournit un guide complet pour créer et utiliser des workflows IA géospatiaux
avec le runner n8n Geospatial. Il inclut des exemples de workflows, des modèles
et des bonnes pratiques pour l'automatisation des traitements géospatiaux.
"""

import json
import os
from typing import Dict, List, Any, Optional
from pathlib import Path

class IAGeospatialWorkflowGuide:
    """
    Guide utilisateur pour la création de workflows IA géospatiaux
    """
    
    def __init__(self):
        self.workflow_templates = {
            "cadastral_analysis": self.cadastral_analysis_template,
            "urban_planning": self.urban_planning_template,
            "environmental_monitoring": self.environmental_monitoring_template,
            "land_registry": self.land_registry_template
        }
    
    def cadastral_analysis_template(self) -> Dict[str, Any]:
        """
        Template pour l'analyse cadastrale automatique
        """
        return {
            "name": "Analyse Cadastrale Automatisée",
            "description": "Workflow pour l'analyse et la validation des données cadastrales",
            "nodes": [
                {
                    "name": "Données d'Entrée",
                    "type": "n8n-nodes-base.manualTrigger",
                    "parameters": {}
                },
                {
                    "name": "Lecture des Données Cadastrales",
                    "type": "n8n-nodes-base.function",
                    "parameters": {
                        "functionCode": """
                        // Charger des données cadastrales à partir d'un fichier ou d'une API
                        const cadastral_data = $input.first.json;
                        return [{json: cadastral_data}];
                        """
                    }
                },
                {
                    "name": "Validation Géométrique",
                    "type": "n8n-nodes-base.function",
                    "parameters": {
                        "functionCode": """
                        // Valider la géométrie des parcelles
                        const parcels = $input.first.json.parcels;
                        const validated_parcels = [];
                        
                        for (const parcel of parcels) {
                            // Vérifier la validité de la géométrie
                            if (isValidGeometry(parcel.geometry)) {
                                validated_parcels.push({...parcel, valid: true});
                            } else {
                                validated_parcels.push({...parcel, valid: false, error: 'Invalid geometry'});
                            }
                        }
                        
                        function isValidGeometry(geom) {
                            // Simulation de validation
                            return geom && geom.coordinates && geom.coordinates.length > 0;
                        }
                        
                        return [{json: {validated_parcels}}];
                        """
                    }
                },
                {
                    "name": "Calcul des Aires",
                    "type": "n8n-nodes-base.function",
                    "parameters": {
                        "functionCode": """
                        // Calculer les aires des parcelles valides
                        const validated_parcels = $input.first.json.validated_parcels;
                        const parcels_with_area = validated_parcels.map(parcel => {
                            if (parcel.valid) {
                                // Calcul de l'aire (simulation)
                                const area = calculateArea(parcel.geometry);
                                return {...parcel, area: area};
                            }
                            return parcel;
                        });
                        
                        function calculateArea(geom) {
                            // Simulation de calcul d'aire
                            return Math.random() * 10000; // Valeur simulée
                        }
                        
                        return [{json: {parcels_with_area}}];
                        """
                    }
                },
                {
                    "name": "Analyse ML des Données",
                    "type": "n8n-nodes-base.function",
                    "parameters": {
                        "functionCode": """
                        // Analyse par machine learning des données cadastrales
                        const data = $input.first.json;
                        const analysis = {
                            ...data,
                            insights: {
                                total_parcels: data.parcels_with_area.length,
                                average_area: data.parcels_with_area.filter(p => p.valid).reduce((sum, p) => sum + (p.area || 0), 0) / data.parcels_with_area.filter(p => p.valid).length,
                                invalid_count: data.parcels_with_area.filter(p => !p.valid).length
                            }
                        };
                        
                        return [{json: analysis}];
                        """
                    }
                },
                {
                    "name": "Export Résultats",
                    "type": "n8n-nodes-base.function",
                    "parameters": {
                        "functionCode": """
                        // Export des résultats
                        const results = $input.first.json;
                        console.log('Analyse cadastrale terminée:', JSON.stringify(results, null, 2));
                        return [{json: results}];
                        """
                    }
                }
            ]
        }
    
    def urban_planning_template(self) -> Dict[str, Any]:
        """
        Template pour la planification urbaine
        """
        return {
            "name": "Planification Urbaine Automatisée",
            "description": "Workflow pour l'analyse spatiale en planification urbaine",
            "nodes": [
                {
                    "name": "Données d'Entrée",
                    "type": "n8n-nodes-base.manualTrigger",
                    "parameters": {}
                },
                {
                    "name": "Chargement des Données Urbaines",
                    "type": "n8n-nodes-base.function",
                    "parameters": {
                        "functionCode": """
                        // Charger des données urbaines
                        const urban_data = $input.first.json;
                        return [{json: urban_data}];
                        """
                    }
                },
                {
                    "name": "Analyse de Voisinage",
                    "type": "n8n-nodes-base.function",
                    "parameters": {
                        "functionCode": """
                        // Analyser les zones de voisinage
                        const data = $input.first.json;
                        
                        // Simulation d'analyse de voisinage
                        const neighborhood_analysis = {
                            ...data,
                            density_analysis: analyzeDensity(data.zones),
                            accessibility_analysis: analyzeAccessibility(data.infrastructure)
                        };
                        
                        function analyzeDensity(zones) {
                            return zones ? zones.map(zone => ({
                                ...zone,
                                density: Math.random() * 100
                            })) : [];
                        }
                        
                        function analyzeAccessibility(infra) {
                            return infra ? infra.map(i => ({
                                ...i,
                                accessibility_score: Math.random()
                            })) : [];
                        }
                        
                        return [{json: neighborhood_analysis}];
                        """
                    }
                },
                {
                    "name": "Prédictions ML",
                    "type": "n8n-nodes-base.function",
                    "parameters": {
                        "functionCode": """
                        // Modèle de prédiction ML
                        const analysis = $input.first.json;
                        const predictions = {
                            ...analysis,
                            growth_predictions: predictGrowth(analysis.density_analysis),
                            development_recommendations: recommendDevelopment(analysis.accessibility_analysis)
                        };
                        
                        function predictGrowth(density_data) {
                            return density_data ? density_data.map(zone => ({
                                zone_id: zone.id,
                                predicted_growth: zone.density * 1.2 + Math.random() * 10
                            })) : [];
                        }
                        
                        function recommendDevelopment(accessibility_data) {
                            return accessibility_data ? accessibility_data.filter(i => i.accessibility_score > 0.7) : [];
                        }
                        
                        return [{json: predictions}];
                        """
                    }
                },
                {
                    "name": "Visualisation",
                    "type": "n8n-nodes-base.function",
                    "parameters": {
                        "functionCode": """
                        // Générer des visualisations
                        const predictions = $input.first.json;
                        const visualization = {
                            map_layers: generateMapLayers(predictions),
                            charts: generateCharts(predictions)
                        };
                        
                        function generateMapLayers(pred) {
                            // Simulation de données cartographiques
                            return {
                                growth_zones: pred.growth_predictions,
                                development_areas: pred.development_recommendations
                            };
                        }
                        
                        function generateCharts(pred) {
                            // Simulation de données pour graphiques
                            return {
                                growth_trend: pred.growth_predictions.map(p => p.predicted_growth),
                                accessibility_score: pred.development_recommendations.map(i => i.accessibility_score)
                            };
                        }
                        
                        return [{json: {predictions, visualization}}];
                        """
                    }
                }
            ]
        }
    
    def environmental_monitoring_template(self) -> Dict[str, Any]:
        """
        Template pour le monitoring environnemental
        """
        return {
            "name": "Monitoring Environnemental Automatisé",
            "description": "Workflow pour le suivi et l'analyse de données environnementales",
            "nodes": [
                {
                    "name": "Données d'Entrée",
                    "type": "n8n-nodes-base.manualTrigger",
                    "parameters": {}
                },
                {
                    "name": "Collecte des Données",
                    "type": "n8n-nodes-base.function",
                    "parameters": {
                        "functionCode": """
                        // Charger des données environnementales
                        const env_data = $input.first.json;
                        return [{json: env_data}];
                        """
                    }
                },
                {
                    "name": "Traitement de Télédétection",
                    "type": "n8n-nodes-base.function",
                    "parameters": {
                        "functionCode": """
                        // Traitement des données de télédétection
                        const data = $input.first.json;
                        const processed = {
                            ...data,
                            ndvi_analysis: calculateNDVI(data.satellite_data),
                            land_use_classification: classifyLandUse(data.satellite_data)
                        };
                        
                        function calculateNDVI(sat_data) {
                            // Simulation de calcul NDVI
                            if (sat_data && sat_data.red && sat_data.nir) {
                                return (sat_data.nir - sat_data.red) / (sat_data.nir + sat_data.red);
                            }
                            return 0;
                        }
                        
                        function classifyLandUse(sat_data) {
                            // Simulation de classification
                            const classes = ['forest', 'water', 'urban', 'agriculture'];
                            return classes[Math.floor(Math.random() * classes.length)];
                        }
                        
                        return [{json: processed}];
                        """
                    }
                },
                {
                    "name": "Analyse Temporelle",
                    "type": "n8n-nodes-base.function",
                    "parameters": {
                        "functionCode": """
                        // Analyse temporelle des changements
                        const processed = $input.first.json;
                        const temporal_analysis = {
                            ...processed,
                            change_detection: detectChanges(processed.time_series),
                            trend_analysis: analyzeTrends(processed.time_series)
                        };
                        
                        function detectChanges(time_series) {
                            // Simulation de détection de changement
                            if (time_series && time_series.length > 1) {
                                return time_series.map((data, i) => ({
                                    date: data.date,
                                    change_detected: Math.abs(data.value - (time_series[i-1]?.value || data.value)) > 0.1
                                }));
                            }
                            return [];
                        }
                        
                        function analyzeTrends(time_series) {
                            // Simulation d'analyse de tendance
                            return {
                                overall_trend: 'increasing', // ou 'decreasing', 'stable'
                                magnitude: 0.5
                            };
                        }
                        
                        return [{json: temporal_analysis}];
                        """
                    }
                },
                {
                    "name": "Alertes IA",
                    "type": "n8n-nodes-base.function",
                    "parameters": {
                        "functionCode": """
                        // Génération d'alertes basées sur l'IA
                        const temporal_data = $input.first.json;
                        const alerts = {
                            ...temporal_data,
                            alerts: generateAlerts(temporal_data.change_detection)
                        };
                        
                        function generateAlerts(changes) {
                            return changes ? changes
                                .filter(change => change.change_detected)
                                .map(change => ({
                                    date: change.date,
                                    type: 'environmental_change',
                                    severity: 'medium',
                                    description: `Changement détecté à la date ${change.date}`
                                })) : [];
                        }
                        
                        return [{json: alerts}];
                        """
                    }
                }
            ]
        }
    
    def land_registry_template(self) -> Dict[str, Any]:
        """
        Template pour le registre foncier
        """
        return {
            "name": "Registre Foncier Automatisé",
            "description": "Workflow pour la gestion automatisée des données foncières",
            "nodes": [
                {
                    "name": "Données d'Entrée",
                    "type": "n8n-nodes-base.manualTrigger",
                    "parameters": {}
                },
                {
                    "name": "Chargement des Actifs Fonciers",
                    "type": "n8n-nodes-base.function",
                    "parameters": {
                        "functionCode": """
                        // Charger des données foncières
                        const land_data = $input.first.json;
                        return [{json: land_data}];
                        """
                    }
                },
                {
                    "name": "Validation des Propriétés",
                    "type": "n8n-nodes-base.function",
                    "parameters": {
                        "functionCode": """
                        // Valider les propriétés foncières
                        const data = $input.first.json;
                        const validated = {
                            ...data,
                            validated_parcels: validateParcels(data.parcels)
                        };
                        
                        function validateParcels(parcels) {
                            if (!parcels) return [];
                            
                            return parcels.map(parcel => {
                                const is_valid = 
                                    parcel.geometry && 
                                    parcel.ownership && 
                                    parcel.area > 0 && 
                                    parcel.zone_type;
                                
                                return {
                                    ...parcel,
                                    valid: is_valid,
                                    validation_errors: is_valid ? [] : ['Données incomplètes']
                                };
                            });
                        }
                        
                        return [{json: validated}];
                        """
                    }
                },
                {
                    "name": "Analyse de Propriété",
                    "type": "n8n-nodes-base.function",
                    "parameters": {
                        "functionCode": """
                        // Analyse des propriétés et des droits
                        const validated = $input.first.json;
                        const property_analysis = {
                            ...validated,
                            ownership_analysis: analyzeOwnership(validated.validated_parcels),
                            value_estimation: estimateValue(validated.validated_parcels)
                        };
                        
                        function analyzeOwnership(parcels) {
                            return parcels ? parcels.map(parcel => ({
                                parcel_id: parcel.id,
                                owner_analysis: {
                                    type: 'individual', // ou 'corporate', 'state'
                                    ownership_stability: Math.random()
                                }
                            })) : [];
                        }
                        
                        function estimateValue(parcels) {
                            return parcels ? parcels.map(parcel => ({
                                parcel_id: parcel.id,
                                estimated_value: (parcel.area || 0) * 100 + Math.random() * 10000
                            })) : [];
                        }
                        
                        return [{json: property_analysis}];
                        """
                    }
                },
                {
                    "name": "Génération de Rapports",
                    "type": "n8n-nodes-base.function",
                    "parameters": {
                        "functionCode": """
                        // Générer des rapports de registre foncier
                        const analysis = $input.first.json;
                        const report = {
                            summary: {
                                total_parcels: analysis.validated_parcels.length,
                                valid_parcels: analysis.validated_parcels.filter(p => p.valid).length,
                                total_value: analysis.value_estimation.reduce((sum, v) => sum + v.estimated_value, 0)
                            },
                            parcels: analysis.validated_parcels,
                            ownership_analysis: analysis.ownership_analysis,
                            value_estimation: analysis.value_estimation
                        };
                        
                        return [{json: report}];
                        """
                    }
                }
            ]
        }
    
    def get_workflow_template(self, template_name: str) -> Optional[Dict[str, Any]]:
        """
        Récupérer un modèle de workflow spécifique
        """
        if template_name in self.workflow_templates:
            return self.workflow_templates[template_name]()
        return None
    
    def list_workflow_templates(self) -> List[str]:
        """
        Lister tous les modèles de workflow disponibles
        """
        return list(self.workflow_templates.keys())
    
    def export_workflow(self, template_name: str, output_path: str) -> bool:
        """
        Exporter un modèle de workflow vers un fichier
        """
        workflow = self.get_workflow_template(template_name)
        if workflow:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(workflow, f, ensure_ascii=False, indent=2)
            return True
        return False


def main():
    """
    Fonction principale pour démontrer l'utilisation du guide
    """
    guide = IAGeospatialWorkflowGuide()
    
    print("Guide Utilisateur pour les Workflows IA Géospatiaux")
    print("=" * 50)
    print(f"Modèles de workflows disponibles: {guide.list_workflow_templates()}")
    
    # Exemple d'export d'un workflow
    if guide.export_workflow("cadastral_analysis", "/qgis-output/cadastral_analysis_workflow.json"):
        print("Modèle de workflow cadastral exporté avec succès")
    
    # Afficher un exemple de workflow
    workflow = guide.get_workflow_template("cadastral_analysis")
    if workflow:
        print(f"\nExemple - {workflow['name']}:")
        print(f"Description: {workflow['description']}")
        print(f"Nombre de noeuds: {len(workflow['nodes'])}")


if __name__ == "__main__":
    main()