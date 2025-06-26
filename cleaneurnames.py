# Script pour détecter la taille d'écran
    st.markdown("""
    <script>
    function updateScreenWidth() {
        const width = window.innerWidth;
        // Envoie la largeur à Streamlit (simulation)
        console.log('Screen width:', width);
    }
    
    window.addEventListener('resize', updateScreenWidth);
    updateScreenWidth();
    </script>
    """, unsafe_allow_html=True)#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script de nettoyage des mots-clés de marque pour SEMRush/Ahrefs
Application Streamlit
"""

import streamlit as st
import pandas as pd
import re
import io
from difflib import SequenceMatcher
from typing import List, Set, Tuple
import base64

# Configuration pour le responsive
st.markdown("""
<style>
    /* Responsive layout */
    @media (max-width: 768px) {
        .stColumns > div {
            min-width: 100% !important;
        }
        .metric-container {
            margin-bottom: 1rem;
        }
    }
    
    /* Style pour les boutons copier */
    .copy-button {
        background: linear-gradient(90deg, #4CAF50, #45a049);
        border: none;
        color: white;
        padding: 8px 16px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 14px;
        margin: 4px 2px;
        cursor: pointer;
        border-radius: 4px;
        transition: all 0.3s;
    }
    
    .copy-button:hover {
        background: linear-gradient(90deg, #45a049, #4CAF50);
        transform: translateY(-2px);
    }
    
    /* Amélioration des zones de texte */
    .stTextArea > div > div > textarea {
        font-family: 'Courier New', monospace;
        font-size: 14px;
    }
    
    /* Responsive metrics */
    .metric-row {
        display: flex;
        flex-wrap: wrap;
        gap: 1rem;
    }
    
    .metric-item {
        flex: 1;
        min-width: 200px;
    }
    
    /* Mobile-first approach */
    .main-container {
        padding: 1rem;
    }
    
    @media (min-width: 768px) {
        .main-container {
            padding: 2rem;
        }
    }
</style>
""", unsafe_allow_html=True)

class BrandKeywordCleaner:
    def __init__(self, similarity_threshold: float = 0.8):
        """
        Initialise le nettoyeur de mots-clés de marque.
        
        Args:
            similarity_threshold: Seuil de similarité (0.0 à 1.0)
        """
        self.similarity_threshold = similarity_threshold
        self.stop_words = {'store', 'shop', 'official', 'france', 'fr', 'com', 'net', 'org', 'eu', 'boutique', 'site'}
        
    def extract_brand_names(self, domains: List[str]) -> Set[str]:
        """
        Extrait les noms de marque à partir des domaines.
        
        Args:
            domains: Liste des domaines à analyser
            
        Returns:
            Set des noms de marque extraits
        """
        brand_names = set()
        
        for domain in domains:
            # Nettoyage du domaine
            clean_domain = domain.lower().strip()
            clean_domain = re.sub(r'^https?://', '', clean_domain)
            clean_domain = re.sub(r'^www\.', '', clean_domain)
            clean_domain = clean_domain.split('.')[0]
            
            # Séparation par tirets et underscores
            parts = re.split(r'[-_]', clean_domain)
            
            for part in parts:
                if len(part) > 2 and part not in self.stop_words:
                    brand_names.add(part)
                    
        return brand_names
    
    def calculate_similarity(self, str1: str, str2: str) -> float:
        """
        Calcule la similarité entre deux chaînes.
        
        Args:
            str1, str2: Chaînes à comparer
            
        Returns:
            Score de similarité (0.0 à 1.0)
        """
        return SequenceMatcher(None, str1.lower(), str2.lower()).ratio()
    
    def is_brand_term(self, keyword: str, brand_names: Set[str]) -> bool:
        """
        Vérifie si un mot-clé EST un terme de marque (mot complet uniquement).
        
        Args:
            keyword: Mot-clé à analyser
            brand_names: Set des noms de marque
            
        Returns:
            True si le mot-clé EST un terme de marque
        """
        keyword_clean = re.sub(r'[^\w\s]', ' ', keyword.lower()).strip()
        
        # Si le keyword est un seul mot, on compare directement
        if len(keyword_clean.split()) == 1:
            for brand in brand_names:
                # Recherche exacte (insensible à la casse)
                if keyword_clean == brand.lower():
                    return True
                
                # Recherche avec similarité pour les fautes d'orthographe
                similarity = self.calculate_similarity(keyword_clean, brand.lower())
                if similarity >= self.similarity_threshold:
                    return True
        
        return False
    
    def load_dataframe(self, uploaded_file) -> List[str]:
        """
        Charge un fichier uploadé et extrait les mots-clés.
        
        Args:
            uploaded_file: Fichier uploadé via Streamlit
            
        Returns:
            Liste des mots-clés
        """
        keywords = []
        
        try:
            # Détection de l'extension
            if uploaded_file.name.lower().endswith('.csv'):
                # Tentative de lecture avec différents encodages
                encodings = ['utf-8', 'utf-16', 'latin-1', 'cp1252', 'iso-8859-1']
                df = None
                
                for encoding in encodings:
                    try:
                        # Reset du pointeur de fichier
                        uploaded_file.seek(0)
                        df = pd.read_csv(uploaded_file, encoding=encoding, sep=None, engine='python')
                        break
                    except (UnicodeDecodeError, UnicodeError):
                        continue
                    except Exception:
                        # Si ce n'est pas un problème d'encodage, on essaie quand même les autres
                        continue
                
                if df is None:
                    # Dernière tentative avec détection automatique
                    uploaded_file.seek(0)
                    df = pd.read_csv(uploaded_file, encoding='utf-8', errors='ignore', sep=None, engine='python')
            else:
                df = pd.read_excel(uploaded_file)
            
            # Vérification que le DataFrame n'est pas vide
            if df is None or df.empty:
                st.error(f"Fichier vide ou illisible: {uploaded_file.name}")
                return keywords
            
            # Recherche de la colonne keyword
            keyword_column = None
            for col in df.columns:
                if 'keyword' in str(col).lower():
                    keyword_column = col
                    break
            
            if keyword_column is None:
                st.error(f"⚠️ Colonne 'keyword' non trouvée dans {uploaded_file.name}")
                st.write(f"Colonnes disponibles: {list(df.columns)}")
                return keywords
            
            # Extraction des mots-clés
            keywords = df[keyword_column].dropna().astype(str).tolist()
            keywords = [kw.strip() for kw in keywords if kw.strip() and kw != 'nan']
            
            st.success(f"{len(keywords)} mots-clés chargés depuis {uploaded_file.name}")
            
        except Exception as e:
            st.error(f"❌ Erreur lors du chargement de {uploaded_file.name}: {str(e)}")
            
        return keywords
    
    def process_files(self, uploaded_files, domains: List[str]) -> Tuple[List[str], dict]:
        """
        Traite les fichiers et détecte les termes de marque.
        
        Args:
            uploaded_files: Liste des fichiers uploadés
            domains: Liste des domaines
            
        Returns:
            Tuple (liste des termes de marque, statistiques)
        """
        with st.spinner("Extraction des noms de marque depuis les domaines..."):
            brand_names = self.extract_brand_names(domains)
            st.write(f"**Noms de marque détectés:** {', '.join(sorted(brand_names))}")
        
        with st.spinner("Chargement des fichiers..."):
            all_keywords = []
            
            for uploaded_file in uploaded_files:
                keywords = self.load_dataframe(uploaded_file)
                all_keywords.extend(keywords)
        
        total_keywords = len(all_keywords)
        st.write(f"**Total:** {total_keywords:,} mots-clés à analyser")
        
        with st.spinner("Détection des termes de marque..."):
            progress_bar = st.progress(0)
            brand_terms = []
            
            for i, keyword in enumerate(all_keywords):
                if self.is_brand_term(keyword, brand_names):
                    brand_terms.append(keyword)
                
                # Mise à jour de la barre de progression
                if i % 100 == 0:
                    progress_bar.progress((i + 1) / len(all_keywords))
            
            progress_bar.progress(1.0)
        
        # Suppression des doublons tout en gardant l'ordre
        unique_terms = list(dict.fromkeys(brand_terms))
        
        stats = {
            'total_keywords': total_keywords,
            'brand_terms_found': len(brand_terms),
            'unique_brand_terms': len(unique_terms),
            'brand_percentage': (len(brand_terms) / total_keywords * 100) if total_keywords > 0 else 0,
            'brand_names': brand_names
        }
        
        return unique_terms, stats
    
    def generate_regex(self, terms: List[str]) -> str:
        """
        Génère une regex à partir des termes de marque.
        
        Args:
            terms: Liste des termes de marque
            
        Returns:
            Regex string
        """
        # Échappement des caractères spéciaux
        escaped_terms = [re.escape(term) for term in terms]
        return f"\\b({'|'.join(escaped_terms)})\\b"
    
    def generate_comma_list(self, terms: List[str]) -> str:
        """
        Génère une liste séparée par des virgules.
        
        Args:
            terms: Liste des termes de marque
            
        Returns:
            String avec termes séparés par des virgules
        """
        return ', '.join(terms)

def copy_to_clipboard_js(text_to_copy: str, button_id: str) -> str:
    """
    Génère le JavaScript pour copier du texte dans le presse-papiers.
    
    Args:
        text_to_copy: Texte à copier
        button_id: ID unique du bouton
        
    Returns:
        HTML avec JavaScript pour la fonction de copie
    """
    # Échappement des caractères spéciaux pour JavaScript
    escaped_text = text_to_copy.replace('\\', '\\\\').replace("'", "\\'").replace('\n', '\\n')
    
    return f"""
    <button class="copy-button" onclick="copyToClipboard_{button_id}()">
        Copier
    </button>
    <script>
    function copyToClipboard_{button_id}() {{
        const text = `{escaped_text}`;
        navigator.clipboard.writeText(text).then(function() {{
            const button = event.target;
            const originalText = button.innerHTML;
            button.innerHTML = 'Copié !';
            button.style.background = 'linear-gradient(90deg, #28a745, #20c997)';
            setTimeout(function() {{
                button.innerHTML = originalText;
                button.style.background = 'linear-gradient(90deg, #4CAF50, #45a049)';
            }}, 2000);
        }}).catch(function(err) {{
            console.error('Erreur lors de la copie: ', err);
            const button = event.target;
            button.innerHTML = 'Erreur';
            button.style.background = 'linear-gradient(90deg, #dc3545, #c82333)';
            setTimeout(function() {{
                button.innerHTML = 'Copier';
                button.style.background = 'linear-gradient(90deg, #4CAF50, #45a049)';
            }}, 2000);
        }});
    }}
    </script>
    """
def create_download_link(content: str, filename: str, text: str) -> str:
    """
    Crée un lien de téléchargement pour du contenu texte.
    
    Args:
        content: Contenu à télécharger
        filename: Nom du fichier
        text: Texte du lien
        
    Returns:
        HTML du lien de téléchargement
    """
    b64 = base64.b64encode(content.encode()).decode()
    return f'<a href="data:text/plain;base64,{b64}" download="{filename}">{text}</a>'

def main():
    """Interface Streamlit principale."""
    
    # Configuration de la page
    st.set_page_config(
        page_title="Nettoyeur de mots-clés de marque",
        page_icon="🔍",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # En-tête
    st.title("Nettoyeur de mots-clés de marque")
    st.markdown("**Extrait automatiquement les termes de marque de vos données SEMRush/Ahrefs**")
    
    # Sidebar pour la configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Domaines
        domains_input = st.text_area(
            "Domaines à analyser",
            placeholder="nike.com\nadidas.fr\npuma-store.com",
            help="Un domaine par ligne ou séparés par des virgules"
        )
        
        # Seuil de similarité
        similarity = st.slider(
            "Seuil de similarité (%)",
            min_value=50,
            max_value=100,
            value=80,
            help="Plus le seuil est bas, plus l'outil détectera de variantes (mais aussi plus de faux positifs)"
        )
        
        st.markdown("---")
        st.markdown("### 📋 Instructions")
        st.markdown("""
        1. **Saisissez vos domaines** (un par ligne)
        2. **Ajustez le seuil** si nécessaire
        3. **Uploadez vos fichiers** Excel/CSV
        4. **Cliquez sur Analyser**
        5. **Copiez les résultats** générés
        """)
    
    # Zone principale
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("📁 Upload des fichiers")
        
        uploaded_files = st.file_uploader(
            "Choisissez vos fichiers SEMRush/Ahrefs",
            type=['xlsx', 'xls', 'csv'],
            accept_multiple_files=True,
            help="Formats acceptés: Excel (.xlsx, .xls) et CSV"
        )
        
        if uploaded_files:
            st.success(f"✅ {len(uploaded_files)} fichier(s) uploadé(s)")
            with st.expander("Fichiers uploadés"):
                for file in uploaded_files:
                    st.write(f"• {file.name} ({file.size:,} bytes)")
    
    with col2:
        st.header("Lancement")
        
        # Validation des entrées
        domains_valid = bool(domains_input.strip())
        files_valid = bool(uploaded_files)
        
        if not domains_valid:
            st.warning("Veuillez saisir au moins un domaine")
        if not files_valid:
            st.warning("Veuillez uploader au moins un fichier")
        
        # Bouton d'analyse
        analyze_btn = st.button(
            "Analyser les fichiers",
            disabled=not (domains_valid and files_valid),
            use_container_width=True
        )
    
    # Traitement
    if analyze_btn and domains_valid and files_valid:
        # Préparation des domaines
        domains = []
        for line in domains_input.replace(',', '\n').split('\n'):
            domain = line.strip()
            if domain:
                domains.append(domain)
        
        # Initialisation du cleaner
        cleaner = BrandKeywordCleaner(similarity_threshold=similarity / 100)
        
        try:
            # Traitement
            terms, stats = cleaner.process_files(uploaded_files, domains)
            
            if not terms:
                st.warning("Aucun terme de marque détecté")
                return
            
            # Affichage des statistiques avec layout responsive
            st.header("Résultats de l'analyse")
            
            # Métriques avec layout adaptatif
            if st.session_state.get('screen_width', 1200) < 768:
                # Mobile : métriques empilées
                st.metric("Mots-clés analysés", f"{stats['total_keywords']:,}")
                st.metric("Termes détectés", f"{stats['brand_terms_found']:,}")
                st.metric("Termes uniques", f"{stats['unique_brand_terms']:,}")
                st.metric("% de marque", f"{stats['brand_percentage']:.1f}%")
            else:
                # Desktop : métriques en ligne
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Mots-clés analysés", f"{stats['total_keywords']:,}")
                with col2:
                    st.metric("Termes détectés", f"{stats['brand_terms_found']:,}")
                with col3:
                    st.metric("Termes uniques", f"{stats['unique_brand_terms']:,}")
                with col4:
                    st.metric("% de marque", f"{stats['brand_percentage']:.1f}%")
            
            # Génération des résultats
            regex = cleaner.generate_regex(terms)
            comma_list = cleaner.generate_comma_list(terms)
            
            # Affichage des résultats
            st.header("Résultats générés")
            
            # Regex avec bouton copier
            st.subheader("Regex générée")
            col_regex_text, col_regex_btn = st.columns([4, 1])
            with col_regex_text:
                st.code(regex, language="regex")
            with col_regex_btn:
                st.markdown(copy_to_clipboard_js(regex, "regex"), unsafe_allow_html=True)
            
            # Liste des termes avec bouton copier
            st.subheader("Liste des termes (séparés par virgules)")
            col_list_text, col_list_btn = st.columns([4, 1])
            with col_list_text:
                st.text_area(
                    "Termes:",
                    value=comma_list,
                    height=100,
                    key="comma_list_display"
                )
            with col_list_btn:
                st.markdown(copy_to_clipboard_js(comma_list, "list"), unsafe_allow_html=True)
            
            # Détail des termes
            with st.expander("Détail des termes détectés", expanded=False):
                detail_list = '\n'.join([f"{i+1:4d}. {term}" for i, term in enumerate(terms)])
                st.text_area(
                    "Tous les termes:",
                    value=detail_list,
                    height=300,
                    key="detail_list_display"
                )
            
            # Téléchargement des résultats
            st.subheader("Téléchargement")
            
            # Préparation du contenu complet
            results_content = f"""RÉSULTATS DE L'ANALYSE DES MOTS-CLÉS DE MARQUE
{'=' * 80}

STATISTIQUES
{'-' * 40}
Mots-clés analysés: {stats['total_keywords']:,}
Termes de marque détectés: {stats['brand_terms_found']:,}
Termes uniques: {stats['unique_brand_terms']:,}
Pourcentage de marque: {stats['brand_percentage']:.1f}%
Noms de marque utilisés: {', '.join(sorted(stats['brand_names']))}
Seuil de similarité: {similarity}%

REGEX GÉNÉRÉE
{'-' * 40}
{regex}

LISTE DES TERMES (séparés par virgules)
{'-' * 40}
{comma_list}

DÉTAIL DES TERMES DÉTECTÉS
{'-' * 40}
{detail_list}
"""
            
            st.download_button(
                label="Télécharger les résultats complets",
                data=results_content,
                file_name="brand_keywords_results.txt",
                mime="text/plain",
                use_container_width=True
            )
            
            st.success("Analyse terminée avec succès !")
            
        except Exception as e:
            st.error(f"Erreur lors de l'analyse: {str(e)}")

if __name__ == "__main__":
    main()
