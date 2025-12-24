# Tutoriel Complet : Utilisation des Agents Géospatiaux IA dans n8n

## Table des Matières
1. [Introduction](#introduction)
2. [Préparation de l'Environnement](#préparation)
3. [Structure des Agents et Fonctionnalités](#structure)
4. [Tutoriel Pratique - Agent Cadastral](#tutoriel-cadastral)
5. [Tutoriel Pratique - Agent Domanial](#tutoriel-domanial)
6. [Tutoriel Pratique - Agent d'Urbanisme](#tutoriel-urbanisme)
7. [Tutoriel Pratique - Agent Environnemental](#tutoriel-environnemental)
8. [Trucs et Astuces](#astuces)
9. [Cas d'Usage Avancés](#cas-avances)

## 1. Introduction {#introduction}

Ce tutoriel vous guidera pas à pas dans l'utilisation des agents géospatiaux IA dans n8n. Vous apprendrez à créer des workflows automatisés pour traiter des données cadastrales, gérer des propriétés domaniales, planifier des aménagements urbains, et surveiller l'environnement.

### Objectifs du Tutoriel
- Comprendre la structure des agents géospatiaux IA
- Créer des workflows fonctionnels pour chaque type d'agent
- Apprendre à intégrer les agents dans des processus automatisés
- Explorer des cas d'usage concrets

## 2. Préparation de l'Environnement {#préparation}

### Étape 1 : Vérification de l'Environnement

Avant de commencer, assurez-vous que votre environnement n8n Geospatial est correctement configuré :

```bash
# Vérifiez que les scripts sont accessibles
ls -la /opt/geoscripts/
# Vous devriez voir :
# cadastral_agent.py
# domain_agent.py  
# other_agents.py
# ia_workflows_user_guide.py
# qgis_processing.py
# postgis_utils.py
# grass_utils.py
# health_check.py
```

### Étape 2 : Création d'un Nouveau Workflow

1. Connectez-vous à votre instance n8n
2. Cliquez sur "Create a workflow"
3. Donnez un nom à votre workflow (ex: "Validation Cadastrale Automatisée")
4. Enregistrez-le

### Étape 3 : Configuration des Variables d'Environnement

Ajoutez ces variables dans votre configuration n8n :
```
N8N_RUNNERS_MODE=external
GDAL_CACHEMAX=1024
GDAL_NUM_THREADS=ALL_CPUS
PROJ_NETWORK=ON
QT_QPA_PLATFORM=offscreen
```

## 3. Structure des Agents et Fonctionnalités {#structure}

Chaque agent dispose de fonctions spécifiques :

### Agent Cadastral (`cadastral_agent.py`)
- `load_parcels()` - Charger les données des parcelles
- `validate_parcels()` - Valider la géométrie et les propriétés
- `detect_cadastral_anomalies()` - Détecter anomalies cadastrales
- `consolidate_parcels()` - Consolider les parcelles adjacentes
- `generate_cadastral_report()` - Générer un rapport complet

### Agent Domanial (`domain_agent.py`)
- `load_domain_properties()` - Charger les propriétés domaniales
- `identify_strategic_zones()` - Identifier zones stratégiques
- `analyze_concessions()` - Analyser les concessions
- `generate_domain_report()` - Générer un rapport domanial

## 4. Tutoriel Pratique - Agent Cadastral {#tutoriel-cadastral}

### Étape 1 : Créer un Workflow de Validation Cadastrale

Commençons par un workflow simple de validation des parcelles cadastrales :

**Noeud 1 : Manual Trigger**
- Cliquez sur "Add Node"
- Sélectionnez "Manual Trigger"
- Cliquez sur "Execute Workflow" pour tester

**Noeud 2 : Données d'entrée (Function)**

Ajoutez un Function Node et collez ce code :

```javascript
// Charger les données d'exemple de parcelles cadastrales
const sampleParcels = {
  "type": "FeatureCollection",
  "features": [
    {
      "type": "Feature",
      "properties": {
        "id": "PAR001",
        "area": 1000,
        "perimeter": 120,
        "owner_id": "OWNER001",
        "zone_type": "urban",
        "current_density": 150
      },
      "geometry": {
        "type": "Polygon",
        "coordinates": [[
          [2.3522, 48.8566],
          [2.3525, 48.8566],
          [2.3525, 48.8570],
          [2.3522, 48.8570],
          [2.3522, 48.8566]
        ]]
      }
    },
    {
      "type": "Feature",
      "properties": {
        "id": "PAR002",
        "area": 800,
        "perimeter": 100,
        "owner_id": "OWNER002",
        "zone_type": "agricultural",
        "current_density": 50
      },
      "geometry": {
        "type": "Polygon",
        "coordinates": [[
          [2.3530, 48.8566],
          [2.3533, 48.8566],
          [2.3533, 48.8570],
          [2.3530, 48.8570],
          [2.3530, 48.8566]
        ]]
      }
    }
  ]
};

return [{json: sampleParcels}];
```

**Noeud 3 : Validation Cadastrale (Function)**

Ajoutez un autre Function Node avec ce code :

```javascript
// Importer et initialiser l'agent cadastral
// Note : Dans un environnement réel, vous utiliseriez Python
// Ici, on simule la validation

const inputData = $input.first.json;
const features = inputData.features;

// Simulation de la validation
const validationResults = features.map(feature => {
  const area = feature.properties.area;
  const geom = feature.geometry;
  
  // Simulation de diverses validations
  const issues = [];
  
  if (area < 100) {
    issues.push({
      type: "area_too_small",
      description: "Superficie trop petite",
      severity: "low"
    });
  }
  
  if (!geom || geom.type !== "Polygon") {
    issues.push({
      type: "geometry_invalid",
      description: "Géométrie invalide",
      severity: "high"
    });
  }
  
  return {
    parcel_id: feature.properties.id,
    issues: issues,
    valid: issues.length === 0
  };
});

return [{
  json: {
    validation_results: validationResults,
    total_parcels: features.length,
    valid_parcels: validationResults.filter(r => r.valid).length,
    invalid_parcels: validationResults.filter(r => !r.valid).length
  }
}];
```

**Noeud 4 : Rapport de Validation (Function)**

Ajoutez un dernier Function Node :

```javascript
const results = $input.first.json;

// Générer un rapport de validation
const report = {
  date: new Date().toISOString(),
  total_parcels: results.total_parcels,
  valid_parcels: results.valid_parcels,
  invalid_parcels: results.invalid_parcels,
  validation_rate: (results.valid_parcels / results.total_parcels * 100).toFixed(2) + "%",
  issues_summary: results.validation_results
    .filter(r => r.issues.length > 0)
    .reduce((acc, curr) => {
      curr.issues.forEach(issue => {
        if (!acc[issue.type]) acc[issue.type] = 0;
        acc[issue.type]++;
      });
      return acc;
    }, {}),
  recommendations: []
};

// Ajouter des recommandations
if (results.invalid_parcels > 0) {
  report.recommendations.push("Corriger les parcelles invalides avant traitement");
}
if (results.valid_parcels < results.total_parcels * 0.9) {
  report.recommendations.push("Taux de validation inférieur à 90%, vérifier les données source");
}

// Sauvegarder dans la base (simulation)
console.log("Validation report generated:", JSON.stringify(report, null, 2));

return [{json: report}];
```

### Étape 2 : Exécuter et Tester le Workflow

1. Cliquez sur "Execute Workflow"
2. Examinez les résultats dans chaque noeud
3. Vérifiez que les validations sont correctement effectuées

### Étape 3 : Extension avec Python (Réel)

Dans un environnement de production, vous utiliserez le script Python réel :

```python
# Dans un HTTP node ou Code node capable d'exécuter Python
import sys
sys.path.append('/opt/geoscripts')

from cadastral_agent import CadastralAgent
import json

# Récupérer les données d'entrée
input_data = {{ $input.first.json }}

# Initialiser l'agent
agent = CadastralAgent()

# Charger les parcelles (convertir les données GeoJSON en GeoDataFrame)
# Ceci nécessite un traitement spécial dans n8n
# La conversion GeoJSON → GeoDataFrame doit être faite en amont

# Simuler le chargement
agent.load_parcels(input_data)

# Exécuter les validations
validation_results = agent.validate_parcels()
anomalies = agent.detect_cadastral_anomalies()

# Générer le rapport
report = agent.generate_cadastral_report()

# Retourner les résultats
return [{
    'json': {
        'validation_results': validation_results,
        'anomalies': anomalies,
        'report': report
    }
}]
```

## 5. Tutoriel Pratique - Agent Domanial {#tutoriel-domanial}

### Étape 1 : Créer un Workflow de Gestion Domaniale

**Noeud 1 : Manual Trigger**

**Noeud 2 : Chargement des Propriétés Domaniales**

```javascript
// Données simulées de propriétés domaniales
const domainProperties = {
  type: "FeatureCollection",
  features: [
    {
      type: "Feature",
      properties: {
        id: "DOM001",
        category: "coastal",
        status: "available",
        area_ha: 50,
        value: 250000,
        management_entity: "state"
      },
      geometry: {
        type: "Polygon",
        coordinates: [[
          [2.3000, 48.8000],
          [2.3005, 48.8000],
          [2.3005, 48.8005],
          [2.3000, 48.8005],
          [2.3000, 48.8000]
        ]]
      }
    },
    {
      type: "Feature",
      properties: {
        id: "DOM002",
        category: "urban_perimeter",
        status: "leased",
        area_ha: 25,
        value: 150000,
        management_entity: "municipality"
      },
      geometry: {
        type: "Polygon",
        coordinates: [[
          [2.3010, 48.8000],
          [2.3015, 48.8000],
          [2.3015, 48.8005],
          [2.3010, 48.8005],
          [2.3010, 48.8000]
        ]]
      }
    }
  ]
};

return [{json: domainProperties}];
```

**Noeud 3 : Analyse des Propriétés Domaniales**

```javascript
const inputData = $input.first.json;
const features = inputData.features;

// Simulation de l'analyse domaniale
const analysis = {
  total_properties: features.length,
  available_properties: 0,
  strategic_properties: 0,
  properties_by_category: {},
  recommendations: []
};

features.forEach(feature => {
  const props = feature.properties;
  
  // Compter par catégorie
  if (!analysis.properties_by_category[props.category]) {
    analysis.properties_by_category[props.category] = 0;
  }
  analysis.properties_by_category[props.category]++;
  
  // Compter les propriétés disponibles
  if (props.status === 'available') {
    analysis.available_properties++;
  }
  
  // Identifier les propriétés stratégiques (simulation)
  if (props.category === 'coastal' || props.category === 'urban_perimeter') {
    analysis.strategic_properties++;
  }
});

// Générer des recommandations
if (analysis.available_properties > 0) {
  analysis.recommendations.push("Explorer les opportunités de valorisation pour les propriétés disponibles");
}

if (analysis.strategic_properties > 0) {
  analysis.recommendations.push("Prioriser le développement des propriétés stratégiques");
}

return [{json: analysis}];
```

**Noeud 4 : Optimisation et Planification**

```javascript
const analysis = $input.first.json;

// Simulation d'opportunités d'optimisation
const optimizationOpportunities = [];

// Simulation de clusters de propriétés
if (analysis.total_properties > 2) {
  optimizationOpportunities.push({
    type: "property_cluster_optimization",
    properties: [1, 2], // ID des propriétés concernées
    recommended_action: "integrated_management",
    estimated_benefit: analysis.total_properties * 10000
  });
}

const optimizationResults = {
  analysis: analysis,
  optimization_opportunities: optimizationOpportunities,
  suggested_actions: [
    "Implémenter une gestion intégrée",
    "Optimiser les revenus des concessions",
    "Évaluer les propriétés sous-exploitées"
  ]
};

return [{json: optimizationResults}];
```

## 6. Tutoriel Pratique - Agent d'Urbanisme {#tutoriel-urbanisme}

### Étape 1 : Analyse de Densité Urbaine

**Noeud 1 : Manual Trigger**

**Noeud 2 : Données Urbaines**

```javascript
const urbanData = {
  zones: [
    {
      id: "ZONE001",
      zone_type: "residential",
      density_limit: 200,
      current_density: 180,
      area_ha: 10,
      population: 1800
    },
    {
      id: "ZONE002", 
      zone_type: "commercial",
      density_limit: 500,
      current_density: 300,
      area_ha: 5,
      population: 500
    },
    {
      id: "ZONE003",
      zone_type: "industrial", 
      density_limit: 100,
      current_density: 80,
      area_ha: 15,
      population: 1200
    }
  ],
  infrastructure: [
    {
      id: "ROAD001",
      type: "primary_road",
      capacity: 2000,
      current_flow: 1800
    },
    {
      id: "TRANSIT001", 
      type: "metro_line",
      capacity: 15000,
      current_flow: 12000
    }
  ]
};

return [{json: urbanData}];
```

**Noeud 3 : Analyse de Densité**

```javascript
const inputData = $input.first.json;
const zones = inputData.zones;

// Analyse de la densité urbaine
const densityAnalysis = {
  overall_urban_density: 0,
  density_by_zone: {},
  overcrowded_zones: [],
  underutilized_zones: [],
  infrastructure_pressure: {}
};

// Calculer la densité moyenne
const totalPopulation = zones.reduce((sum, zone) => sum + zone.population, 0);
const totalArea = zones.reduce((sum, zone) => sum + zone.area_ha, 0);
densityAnalysis.overall_urban_density = totalPopulation / totalArea;

// Analyse par type de zone
zones.forEach(zone => {
  const zoneDensity = zone.current_density;
  const utilizationRate = zone.current_density / zone.density_limit;
  
  densityAnalysis.density_by_zone[zone.id] = {
    type: zone.zone_type,
    current_density: zoneDensity,
    utilization_rate: utilizationRate,
    population: zone.population,
    area: zone.area_ha
  };
  
  // Identifier zones surpeuplées ou sous-utilisées
  if (utilizationRate > 0.9) {
    densityAnalysis.overcrowded_zones.push(zone.id);
  } else if (utilizationRate < 0.5 && zone.zone_type !== 'industrial') {
    densityAnalysis.underutilized_zones.push(zone.id);
  }
});

return [{json: densityAnalysis}];
```

## 7. Tutoriel Pratique - Agent Environnemental {#tutoriel-environnemental}

### Étape 1 : Surveillance Environnementale

**Noeud 1 : Manual Trigger**

**Noeud 2 : Données Environnementales**

```javascript
const environmentalData = {
  monitoring_stations: [
    {
      id: "STATION001",
      coordinates: [2.3522, 48.8566],
      air_quality: 0.7,
      water_quality: 0.8,
      soil_quality: 0.6,
      overall_score: 0.7
    },
    {
      id: "STATION002",
      coordinates: [2.3500, 48.8500], 
      air_quality: 0.4,
      water_quality: 0.5,
      soil_quality: 0.3,
      overall_score: 0.4
    }
  ],
  risk_zones: [
    {
      id: "RISK001",
      type: "flood",
      probability: 0.3,
      impact_level: "high",
      area_ha: 50
    },
    {
      id: "RISK002",
      type: "pollution",
      probability: 0.7,
      impact_level: "medium", 
      area_ha: 25
    }
  ],
  biodiversity_sites: [
    {
      id: "BIO001",
      species_richness: 120,
      endemic_species: 15,
      conservation_status: "protected"
    }
  ]
};

return [{json: environmentalData}];
```

**Noeud 3 : Analyse de Qualité Environnementale**

```javascript
const inputData = $input.first.json;

// Analyse de la qualité environnementale
const qualityAssessment = {
  overall_quality_index: 0,
  quality_by_parameter: {},
  pollution_hotspots: [],
  conservation_priorities: []
};

// Calculer l'index global
const avgScore = inputData.monitoring_stations.reduce((sum, station) => 
  sum + station.overall_score, 0) / inputData.monitoring_stations.length;
qualityAssessment.overall_quality_index = avgScore;

// Analyse par paramètre
const airQuality = inputData.monitoring_stations.reduce((sum, station) => 
  sum + station.air_quality, 0) / inputData.monitoring_stations.length;
const waterQuality = inputData.monitoring_stations.reduce((sum, station) => 
  sum + station.water_quality, 0) / inputData.monitoring_stations.length;

qualityAssessment.quality_by_parameter = {
  air_quality: airQuality,
  water_quality: waterQuality,
  soil_quality: inputData.monitoring_stations.reduce((sum, station) => 
    sum + station.soil_quality, 0) / inputData.monitoring_stations.length
};

// Identifier les points chauds de pollution
inputData.monitoring_stations.forEach(station => {
  if (station.overall_score < 0.5) {
    qualityAssessment.pollution_hotspots.push({
      station_id: station.id,
      coordinates: station.coordinates,
      quality_score: station.overall_score,
      severity: station.overall_score < 0.3 ? "high" : "medium"
    });
  }
});

// Priorités de conservation
inputData.biodiversity_sites.forEach(site => {
  if (site.conservation_status === "protected" && site.endemic_species > 10) {
    qualityAssessment.conservation_priorities.push({
      site_id: site.id,
      priority: "high",
      endemic_species: site.endemic_species,
      species_richness: site.species_richness
    });
  }
});

return [{json: qualityAssessment}];
```

## 8. Trucs et Astuces {#astuces}

### Astuce 1 : Utilisation des Expressions n8n

Dans vos Function Nodes, utilisez les expressions n8n pour manipuler les données :

```javascript
// Accéder aux données du précédent noeud
const previousData = $input.first.json;

// Accéder à des variables spécifiques
const specificValue = $input.first.json.specificField;

// Manipuler les dates
const now = new Date().toISOString();

// Boucler sur des collections
const processed = previousData.features.map(feature => ({
  ...feature,
  processed: true
}));
```

### Astuce 2 : Gestion des Erreurs

Toujours inclure une gestion des erreurs :

```javascript
try {
  // Votre code de traitement
  const result = processData($input.first.json);
  return [{json: result}];
} catch (error) {
  console.error("Erreur dans le traitement:", error);
  return [{
    json: {
      error: true,
      message: error.message,
      timestamp: new Date().toISOString()
    }
  }];
}
```

### Astuce 3 : Débogage

Utilisez des Debug Nodes pour inspecter les données à chaque étape :

```javascript
// Dans un Function Node de débogage
console.log("Données d'entrée:", JSON.stringify($input.first.json, null, 2));
return $input.all(); // Passer les données inchangées
```

## 9. Cas d'Usage Avancés {#cas-avances}

### Cas d'Usage 1 : Workflow de Suivi des Concessions Domaniales

```javascript
// Workflow complet de suivi des concessions
const workflow = {
  trigger: "cron_30_days", // Exécution tous les 30 jours
  
  nodes: [
    {
      name: "Load Concessions",
      type: "function",
      code: `
        // Chargement des concessions depuis la base
        const concessions = await loadConcessionsFromDB();
        return [{json: concessions}];
      `
    },
    {
      name: "Check Expirations",
      type: "function", 
      code: `
        const concessions = $input.first.json;
        const soonToExpire = concessions.filter(c => 
          new Date(c.endDate) < new Date(Date.now() + 90*24*60*60*1000) // 90 jours
        );
        return [{json: soonToExpire}];
      `
    },
    {
      name: "Send Notifications",
      type: "email",
      config: {
        to: "{{item.email}}",
        subject: "Rappel : Concession expirant bientôt",
        body: "Votre concession {{item.id}} expire le {{item.endDate}}"
      }
    }
  ]
};
```

### Cas d'Usage 2 : Workflow de Surveillance Cadastrale en Temps Réel

```json
{
  "nodes": [
    {
      "name": "Webhook Trigger",
      "type": "n8n-nodes-base.webhook",
      "parameters": {
        "path": "cadastral-update",
        "responseMode": "responseNode"
      }
    },
    {
      "name": "Validate Parcels",
      "type": "n8n-nodes-base.function",
      "parameters": {
        "functionCode": "// Code de validation cadastrale"
      }
    },
    {
      "name": "Store Results", 
      "type": "n8n-nodes-base.httpRequest",
      "parameters": {
        "url": "http://your-db-api/cadastral-results",
        "method": "POST"
      }
    }
  ]
}
```

### Cas d'Usage 3 : Workflow Multi-Agent pour Planification Intégrée

```javascript
// Workflow combinant plusieurs agents
const integratedPlanningWorkflow = {
  nodes: [
    {
      // Collecte des données géospatiales
    },
    {
      // Analyse cadastrale
    },
    {
      // Analyse domaniale  
    },
    {
      // Analyse urbaine
    },
    {
      // Synthèse et recommandations
    }
  ]
};
```

## Conclusion

Ce tutoriel complet vous a montré comment utiliser les agents géospatiaux IA dans n8n pour automatiser des processus complexes. Les agents cadastral, domanial, d'urbanisme et environnemental offrent des capacités d'analyse avancées combinant géomatique et intelligence artificielle.

Les workflows peuvent être adaptés à vos besoins spécifiques en combinant les différents agents et en ajoutant des étapes de validation, de notification et d'analyse selon vos exigences.