-- ============================================================================
-- EXTENSION DU SCHÉMA POSTGIS - AMÉLIORATIONS PROFESSIONNELLES
-- Version: 2.0
-- Description: Ajout des tables métier pour gestion domaniale professionnelle
-- ============================================================================

-- ============================================================================
-- SCHEMA: domaine - Tables d'évaluation et documentation
-- ============================================================================

-- Table des évaluations de biens (historique des estimations)
CREATE TABLE IF NOT EXISTS domaine.evaluations (
    id SERIAL PRIMARY KEY,
    bien_id INTEGER NOT NULL REFERENCES domaine.biens(id) ON DELETE CASCADE,
    date_evaluation DATE NOT NULL DEFAULT CURRENT_DATE,
    type_evaluation VARCHAR(50) NOT NULL CHECK (type_evaluation IN ('venale', 'locative', 'domaniale', 'expertise')),
    valeur_venale NUMERIC(15,2),
    valeur_locative_annuelle NUMERIC(15,2),
    valeur_locative_mensuelle NUMERIC(15,2) GENERATED ALWAYS AS (valeur_locative_annuelle / 12) STORED,
    methode_evaluation VARCHAR(100),
    source_donnees VARCHAR(255),
    evaluateur VARCHAR(255),
    organisme_evaluateur VARCHAR(255),
    reference_rapport VARCHAR(100),
    justification TEXT,
    donnees_dvf JSONB,  -- Données DVF associées
    date_creation TIMESTAMP DEFAULT NOW(),
    date_modification TIMESTAMP DEFAULT NOW()
);

-- Table des documents associés aux biens
CREATE TABLE IF NOT EXISTS domaine.documents (
    id SERIAL PRIMARY KEY,
    bien_id INTEGER NOT NULL REFERENCES domaine.biens(id) ON DELETE CASCADE,
    type_document VARCHAR(100) NOT NULL CHECK (type_document IN (
        'acte_propriete', 'bail', 'convention', 'arrete', 'pv_reception',
        'plan_cadastral', 'plan_masse', 'photo', 'rapport_expertise',
        'attestation', 'diagnostic', 'facture', 'courrier', 'autre'
    )),
    titre VARCHAR(255) NOT NULL,
    description TEXT,
    chemin_fichier VARCHAR(500),
    nom_fichier_original VARCHAR(255),
    taille_octets BIGINT,
    mime_type VARCHAR(100),
    hash_sha256 VARCHAR(64),
    metadata JSONB,
    date_document DATE,
    uploaded_by VARCHAR(255),
    date_upload TIMESTAMP DEFAULT NOW()
);

-- Table des références DVF pour le calcul de valeur
CREATE TABLE IF NOT EXISTS domaine.references_dvf (
    id SERIAL PRIMARY KEY,
    id_mutation VARCHAR(50) UNIQUE,
    date_mutation DATE,
    nature_mutation VARCHAR(100),
    valeur_fonciere NUMERIC(15,2),
    code_commune VARCHAR(5),
    code_postal VARCHAR(5),
    nom_commune VARCHAR(255),
    adresse_nom_voie VARCHAR(255),
    type_local VARCHAR(100),
    surface_reelle_bati NUMERIC,
    nombre_pieces INTEGER,
    surface_terrain NUMERIC,
    code_nature_culture VARCHAR(10),
    geom GEOMETRY(Point, 2154),
    donnees_brutes JSONB,
    date_import TIMESTAMP DEFAULT NOW()
);

-- ============================================================================
-- SCHEMA: traitement - Tables de workflow et alertes
-- ============================================================================

-- Table de suivi des dossiers d'instruction
CREATE TABLE IF NOT EXISTS traitement.dossiers (
    id SERIAL PRIMARY KEY,
    reference VARCHAR(50) UNIQUE NOT NULL,
    type_dossier VARCHAR(100) NOT NULL CHECK (type_dossier IN (
        'acquisition', 'cession', 'occupation', 'regularisation',
        'contentieux', 'evaluation', 'controle', 'autre'
    )),
    statut VARCHAR(50) NOT NULL DEFAULT 'nouveau' CHECK (statut IN (
        'nouveau', 'en_instruction', 'en_attente', 'valide', 'refuse', 'cloture', 'archive'
    )),
    priorite VARCHAR(20) DEFAULT 'normale' CHECK (priorite IN ('basse', 'normale', 'haute', 'urgente')),
    bien_id INTEGER REFERENCES domaine.biens(id),
    demandeur_nom VARCHAR(255),
    demandeur_contact VARCHAR(255),
    demandeur_telephone VARCHAR(20),
    objet TEXT,
    description TEXT,
    instructeur VARCHAR(255),
    date_reception DATE DEFAULT CURRENT_DATE,
    date_limite DATE,
    date_decision DATE,
    decision VARCHAR(100),
    motif_decision TEXT,
    observations TEXT,
    donnees_instruction JSONB,
    workflow_id VARCHAR(100),
    execution_id VARCHAR(100),
    date_creation TIMESTAMP DEFAULT NOW(),
    date_modification TIMESTAMP DEFAULT NOW()
);

-- Table des alertes et notifications
CREATE TABLE IF NOT EXISTS traitement.alertes (
    id SERIAL PRIMARY KEY,
    type_alerte VARCHAR(100) NOT NULL CHECK (type_alerte IN (
        'modification_cadastrale', 'echeance_bail', 'controle_occupation',
        'evaluation_requise', 'document_manquant', 'dossier_urgent',
        'anomalie_detectee', 'rappel', 'autre'
    )),
    priorite VARCHAR(20) NOT NULL DEFAULT 'normale' CHECK (priorite IN ('info', 'normale', 'importante', 'critique')),
    titre VARCHAR(255) NOT NULL,
    message TEXT NOT NULL,
    bien_id INTEGER REFERENCES domaine.biens(id),
    dossier_id INTEGER REFERENCES traitement.dossiers(id),
    destinataire VARCHAR(255),
    canal_notification VARCHAR(20) NOT NULL DEFAULT 'sms' CHECK (canal_notification IN ('email', 'sms', 'webhook', 'push')),
    telephone_destinataire VARCHAR(20),
    email_destinataire VARCHAR(255),
    webhook_url VARCHAR(500),
    statut_envoi VARCHAR(20) DEFAULT 'en_attente' CHECK (statut_envoi IN ('en_attente', 'envoye', 'echec', 'lu')),
    date_envoi TIMESTAMP,
    date_lecture TIMESTAMP,
    erreur_envoi TEXT,
    donnees_supplementaires JSONB,
    date_creation TIMESTAMP DEFAULT NOW()
);

-- Table de configuration SMS
CREATE TABLE IF NOT EXISTS traitement.config_sms (
    id SERIAL PRIMARY KEY,
    provider VARCHAR(50) NOT NULL DEFAULT 'twilio' CHECK (provider IN ('twilio', 'nexmo', 'ovh', 'orange')),
    api_endpoint VARCHAR(255),
    api_key_encrypted VARCHAR(500),
    sender_id VARCHAR(20),
    actif BOOLEAN DEFAULT TRUE,
    config_json JSONB,
    date_creation TIMESTAMP DEFAULT NOW()
);

-- ============================================================================
-- INDEX POUR PERFORMANCES
-- ============================================================================

-- Index sur evaluations
CREATE INDEX IF NOT EXISTS idx_evaluations_bien ON domaine.evaluations(bien_id);
CREATE INDEX IF NOT EXISTS idx_evaluations_date ON domaine.evaluations(date_evaluation);
CREATE INDEX IF NOT EXISTS idx_evaluations_type ON domaine.evaluations(type_evaluation);

-- Index sur documents
CREATE INDEX IF NOT EXISTS idx_documents_bien ON domaine.documents(bien_id);
CREATE INDEX IF NOT EXISTS idx_documents_type ON domaine.documents(type_document);

-- Index sur références DVF
CREATE INDEX IF NOT EXISTS idx_dvf_commune ON domaine.references_dvf(code_commune);
CREATE INDEX IF NOT EXISTS idx_dvf_date ON domaine.references_dvf(date_mutation);
CREATE INDEX IF NOT EXISTS idx_dvf_geom ON domaine.references_dvf USING GIST(geom);

-- Index sur dossiers
CREATE INDEX IF NOT EXISTS idx_dossiers_bien ON traitement.dossiers(bien_id);
CREATE INDEX IF NOT EXISTS idx_dossiers_statut ON traitement.dossiers(statut);
CREATE INDEX IF NOT EXISTS idx_dossiers_type ON traitement.dossiers(type_dossier);
CREATE INDEX IF NOT EXISTS idx_dossiers_date_limite ON traitement.dossiers(date_limite);

-- Index sur alertes
CREATE INDEX IF NOT EXISTS idx_alertes_bien ON traitement.alertes(bien_id);
CREATE INDEX IF NOT EXISTS idx_alertes_statut ON traitement.alertes(statut_envoi);
CREATE INDEX IF NOT EXISTS idx_alertes_date ON traitement.alertes(date_creation);
CREATE INDEX IF NOT EXISTS idx_alertes_priorite ON traitement.alertes(priorite);

-- ============================================================================
-- TRIGGERS DE MISE À JOUR AUTOMATIQUE
-- ============================================================================

-- Trigger pour evaluations
DROP TRIGGER IF EXISTS trigger_evaluations_modification ON domaine.evaluations;
CREATE TRIGGER trigger_evaluations_modification
    BEFORE UPDATE ON domaine.evaluations
    FOR EACH ROW
    EXECUTE FUNCTION update_date_modification();

-- Trigger pour dossiers
DROP TRIGGER IF EXISTS trigger_dossiers_modification ON traitement.dossiers;
CREATE TRIGGER trigger_dossiers_modification
    BEFORE UPDATE ON traitement.dossiers
    FOR EACH ROW
    EXECUTE FUNCTION update_date_modification();

-- ============================================================================
-- VUES MÉTIER
-- ============================================================================

-- Vue des biens avec dernière évaluation
CREATE OR REPLACE VIEW domaine.v_biens_evaluations AS
SELECT 
    b.*,
    e.date_evaluation as derniere_evaluation_date,
    e.valeur_venale as derniere_valeur_venale,
    e.valeur_locative_annuelle as derniere_valeur_locative,
    e.evaluateur as dernier_evaluateur
FROM domaine.biens b
LEFT JOIN LATERAL (
    SELECT * FROM domaine.evaluations 
    WHERE bien_id = b.id 
    ORDER BY date_evaluation DESC 
    LIMIT 1
) e ON true;

-- Vue des dossiers en cours avec deadline
CREATE OR REPLACE VIEW traitement.v_dossiers_urgents AS
SELECT 
    d.*,
    b.reference as bien_reference,
    b.designation as bien_designation,
    d.date_limite - CURRENT_DATE as jours_restants
FROM traitement.dossiers d
LEFT JOIN domaine.biens b ON d.bien_id = b.id
WHERE d.statut NOT IN ('cloture', 'archive')
  AND d.date_limite IS NOT NULL
  AND d.date_limite <= CURRENT_DATE + INTERVAL '7 days'
ORDER BY d.date_limite;

-- Vue des alertes non lues
CREATE OR REPLACE VIEW traitement.v_alertes_pending AS
SELECT 
    a.*,
    b.reference as bien_reference,
    d.reference as dossier_reference
FROM traitement.alertes a
LEFT JOIN domaine.biens b ON a.bien_id = b.id
LEFT JOIN traitement.dossiers d ON a.dossier_id = d.id
WHERE a.statut_envoi IN ('en_attente', 'echec')
ORDER BY 
    CASE a.priorite 
        WHEN 'critique' THEN 1 
        WHEN 'importante' THEN 2 
        WHEN 'normale' THEN 3 
        ELSE 4 
    END,
    a.date_creation;

-- Vue statistiques DVF par commune
CREATE OR REPLACE VIEW domaine.v_stats_dvf_commune AS
SELECT 
    code_commune,
    nom_commune,
    COUNT(*) as nb_transactions,
    AVG(valeur_fonciere) as valeur_moyenne,
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY valeur_fonciere) as valeur_mediane,
    MIN(valeur_fonciere) as valeur_min,
    MAX(valeur_fonciere) as valeur_max,
    AVG(CASE WHEN surface_terrain > 0 THEN valeur_fonciere / surface_terrain END) as prix_m2_terrain_moyen,
    AVG(CASE WHEN surface_reelle_bati > 0 THEN valeur_fonciere / surface_reelle_bati END) as prix_m2_bati_moyen,
    MAX(date_mutation) as derniere_transaction
FROM domaine.references_dvf
WHERE valeur_fonciere > 0
GROUP BY code_commune, nom_commune;

-- ============================================================================
-- FONCTIONS MÉTIER
-- ============================================================================

-- Fonction de calcul de valeur locative estimée basée sur DVF
CREATE OR REPLACE FUNCTION domaine.calculer_valeur_locative(
    p_bien_id INTEGER,
    p_rayon_metres INTEGER DEFAULT 1000
)
RETURNS TABLE (
    valeur_locative_estimee NUMERIC,
    valeur_venale_estimee NUMERIC,
    nb_references INTEGER,
    prix_m2_moyen NUMERIC,
    confiance VARCHAR
) AS $$
DECLARE
    v_geom GEOMETRY;
    v_surface NUMERIC;
BEGIN
    -- Récupérer la géométrie et surface du bien
    SELECT geom, surface INTO v_geom, v_surface
    FROM domaine.biens WHERE id = p_bien_id;
    
    IF v_geom IS NULL THEN
        RETURN QUERY SELECT NULL::NUMERIC, NULL::NUMERIC, 0, NULL::NUMERIC, 'aucune_donnee'::VARCHAR;
        RETURN;
    END IF;
    
    -- Calculer basé sur les transactions DVF proches
    RETURN QUERY
    SELECT 
        -- Valeur locative = ~5% de la valeur vénale annuellement
        ROUND(AVG(dvf.valeur_fonciere / NULLIF(dvf.surface_terrain, 0)) * COALESCE(v_surface, 1) * 0.05, 2) as valeur_locative,
        ROUND(AVG(dvf.valeur_fonciere / NULLIF(dvf.surface_terrain, 0)) * COALESCE(v_surface, 1), 2) as valeur_venale,
        COUNT(*)::INTEGER as nb_refs,
        ROUND(AVG(dvf.valeur_fonciere / NULLIF(dvf.surface_terrain, 0)), 2) as prix_m2,
        CASE 
            WHEN COUNT(*) >= 10 THEN 'haute'
            WHEN COUNT(*) >= 5 THEN 'moyenne'
            WHEN COUNT(*) >= 1 THEN 'faible'
            ELSE 'aucune_donnee'
        END::VARCHAR as conf
    FROM domaine.references_dvf dvf
    WHERE ST_DWithin(dvf.geom, v_geom, p_rayon_metres)
      AND dvf.date_mutation >= CURRENT_DATE - INTERVAL '3 years'
      AND dvf.valeur_fonciere > 0
      AND dvf.surface_terrain > 0;
END;
$$ LANGUAGE plpgsql;

-- Fonction pour créer une alerte SMS
CREATE OR REPLACE FUNCTION traitement.creer_alerte_sms(
    p_type VARCHAR,
    p_titre VARCHAR,
    p_message TEXT,
    p_telephone VARCHAR,
    p_priorite VARCHAR DEFAULT 'normale',
    p_bien_id INTEGER DEFAULT NULL
)
RETURNS INTEGER AS $$
DECLARE
    v_alerte_id INTEGER;
BEGIN
    INSERT INTO traitement.alertes (
        type_alerte, priorite, titre, message, 
        bien_id, canal_notification, telephone_destinataire
    ) VALUES (
        p_type, p_priorite, p_titre, p_message,
        p_bien_id, 'sms', p_telephone
    )
    RETURNING id INTO v_alerte_id;
    
    RETURN v_alerte_id;
END;
$$ LANGUAGE plpgsql;

-- ============================================================================
-- PERMISSIONS
-- ============================================================================

GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA domaine TO geo;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA traitement TO geo;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA domaine TO geo;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA traitement TO geo;
GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA domaine TO geo;
GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA traitement TO geo;

-- ============================================================================
-- DONNÉES D'INITIALISATION
-- ============================================================================

-- Configuration SMS par défaut (Twilio)
INSERT INTO traitement.config_sms (provider, api_endpoint, sender_id, config_json)
VALUES ('twilio', 'https://api.twilio.com/2010-04-01', 'CADASTRE', '{"account_sid": "", "auth_token": ""}')
ON CONFLICT DO NOTHING;
