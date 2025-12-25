-- Création des extensions
CREATE EXTENSION IF NOT EXISTS postgis;
CREATE EXTENSION IF NOT EXISTS postgis_topology;
CREATE EXTENSION IF NOT EXISTS pg_trgm;
CREATE EXTENSION IF NOT EXISTS unaccent;

-- Schémas
CREATE SCHEMA IF NOT EXISTS cadastre;
CREATE SCHEMA IF NOT EXISTS domaine;
CREATE SCHEMA IF NOT EXISTS traitement;

-- Tables principales cadastrales
CREATE TABLE IF NOT EXISTS cadastre.parcelles (
    id SERIAL PRIMARY KEY,
    id_parcelle VARCHAR(20) UNIQUE NOT NULL,
    code_commune VARCHAR(5) NOT NULL,
    prefixe VARCHAR(3),
    section VARCHAR(2) NOT NULL,
    numero VARCHAR(4) NOT NULL,
    contenance NUMERIC,
    arpente BOOLEAN DEFAULT FALSE,
    date_import TIMESTAMP DEFAULT NOW(),
    geom GEOMETRY(MultiPolygon, 2154)
);

CREATE TABLE IF NOT EXISTS cadastre.proprietaires (
    id SERIAL PRIMARY KEY,
    id_parcelle VARCHAR(20) REFERENCES cadastre.parcelles(id_parcelle),
    compte VARCHAR(20),
    nom VARCHAR(255),
    prenom VARCHAR(255),
    adresse TEXT,
    code_droit VARCHAR(2),
    date_import TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS cadastre.batiments (
    id SERIAL PRIMARY KEY,
    id_batiment VARCHAR(30) UNIQUE,
    id_parcelle VARCHAR(20),
    type VARCHAR(50),
    date_construction INTEGER,
    surface_sol NUMERIC,
    hauteur NUMERIC,
    geom GEOMETRY(MultiPolygon, 2154)
);

-- Tables domaniales
CREATE TABLE IF NOT EXISTS domaine.biens (
    id SERIAL PRIMARY KEY,
    reference VARCHAR(50) UNIQUE NOT NULL,
    type_bien VARCHAR(100),
    designation TEXT,
    adresse TEXT,
    code_commune VARCHAR(5),
    surface NUMERIC,
    valeur_venale NUMERIC,
    date_acquisition DATE,
    origine_propriete TEXT,
    affectation VARCHAR(100),
    gestionnaire VARCHAR(255),
    statut VARCHAR(50) DEFAULT 'actif',
    observations TEXT,
    date_creation TIMESTAMP DEFAULT NOW(),
    date_modification TIMESTAMP DEFAULT NOW(),
    geom GEOMETRY(MultiPolygon, 2154)
);

CREATE TABLE IF NOT EXISTS domaine.occupations (
    id SERIAL PRIMARY KEY,
    bien_id INTEGER REFERENCES domaine.biens(id),
    occupant VARCHAR(255),
    type_occupation VARCHAR(100),
    date_debut DATE,
    date_fin DATE,
    loyer NUMERIC,
    observations TEXT
);

-- Table de suivi des traitements
CREATE TABLE IF NOT EXISTS traitement.logs (
    id SERIAL PRIMARY KEY,
    workflow_id VARCHAR(100),
    execution_id VARCHAR(100),
    etape VARCHAR(100),
    statut VARCHAR(20),
    message TEXT,
    donnees JSONB,
    date_execution TIMESTAMP DEFAULT NOW()
);

-- Index spatiaux
CREATE INDEX IF NOT EXISTS idx_parcelles_geom ON cadastre.parcelles USING GIST(geom);
CREATE INDEX IF NOT EXISTS idx_batiments_geom ON cadastre.batiments USING GIST(geom);
CREATE INDEX IF NOT EXISTS idx_biens_geom ON domaine.biens USING GIST(geom);

-- Index attributaires
CREATE INDEX IF NOT EXISTS idx_parcelles_commune ON cadastre.parcelles(code_commune);
CREATE INDEX IF NOT EXISTS idx_parcelles_section ON cadastre.parcelles(section);
CREATE INDEX IF NOT EXISTS idx_biens_commune ON domaine.biens(code_commune);
CREATE INDEX IF NOT EXISTS idx_biens_type ON domaine.biens(type_bien);

-- Fonction de mise à jour automatique de date_modification
CREATE OR REPLACE FUNCTION update_date_modification()
RETURNS TRIGGER AS $$
BEGIN
    NEW.date_modification = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trigger_biens_modification ON domaine.biens;
CREATE TRIGGER trigger_biens_modification
    BEFORE UPDATE ON domaine.biens
    FOR EACH ROW
    EXECUTE FUNCTION update_date_modification();

-- Vue pour jointure parcelles-biens
CREATE OR REPLACE VIEW domaine.v_biens_parcelles AS
SELECT 
    b.*,
    p.section,
    p.numero,
    p.contenance as contenance_cadastrale,
    p.geom as geom_parcelle
FROM domaine.biens b
LEFT JOIN cadastre.parcelles p ON ST_Intersects(b.geom, p.geom);

-- Permissions
GRANT USAGE ON SCHEMA cadastre TO geo;
GRANT USAGE ON SCHEMA domaine TO geo;
GRANT USAGE ON SCHEMA traitement TO geo;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA cadastre TO geo;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA domaine TO geo;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA traitement TO geo;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA cadastre TO geo;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA domaine TO geo;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA traitement TO geo;
