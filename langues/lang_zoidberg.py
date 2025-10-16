LANG_ZOIDBERG = {
    "fr": {
        "footer": "© 2025 - Portfolio personnel développé avec Streamlit",
        "proj_zoidberg": "🩻 Classification de scans pulmonaires",
        "proj_zoidberg_description": "**Objectif :** prédire si un scan de poumon est sain ou malade.",

        "metrics_title": "Métriques de performance du meilleur modèle",
        "metrics_rows": [
            [
                {"label": "F1-Score Global", "value": "80%", "help": "Le F1-score combine précision et rappel pour mesurer la performance globale."},
                {"label": "Pour les sains", "value": "92%"},
                {"label": "Pour la pneumonie bactérienne", "value": "83%"},
                {"label": "Pour la pneumonie virale", "value": "62%"},
            ],
            [
                {"label": "AUC Global", "value": "91.9%", "help": "L’AUC mesure la capacité du modèle à distinguer correctement les classes."},
                {"label": "Pour les sains", "value": "98.6%"},
                {"label": "Pour la pneumonie bactérienne", "value": "91.0%"},
                {"label": "Pour la pneumonie virale", "value": "86.1%"},
            ]
        ],

        "preprocess_data": "Préprocessing des données",
        "example_scans_title": "**Exemples de scans**",
        "example_scans": [
            {"caption": "Scan sain", "path": "../projets/zoidberg/images/Scan_Sain_Exemple.jpeg"},
            {"caption": "Pneumonie bactérienne", "path": "../projets/zoidberg/images/Scan_PneumonieBacterienne.jpeg"},
            {"caption": "Pneumonie virale", "path": "../projets/zoidberg/images/Scan_PneumonieViral.jpeg"},
        ],

        "dataset_title": "**Statistiques du dataset**",
        "dataset": {
            "Classe": ["BACTERIA", "NORMAL", "VIRUS"],
            "Nombre d'images": [2780, 1585, 1493],
            "Poids": [0.70, 1.23, 1.30]
        },
        "dataset_image_caption": "Distribution des classes",

        "preprocessing_title": "**Étapes de nettoyage et de préparation**",
        "preprocessing_steps": [
            {"text": "- Normalisation avec StandardScaler pour rendre les images comparables", "image": ["../projets/zoidberg/images/apresScaler.png", "../projets/zoidberg/images/avantScaler.png"], "caption": ""},
            {"text": "- Redimensionnement des images pour homogénéiser les dimensions", "image": ["../projets/zoidberg/images/imageweight.png"], "caption": "Histogramme des dimensions"},
            {"text": "- Augmentation des données par flip horizontal (utile surtout pour la classe virale)", "image": ["../projets/zoidberg/images/flipimage.png"], "caption": "Exemple de flip horizontal"},
        ],

        "models_title": "Modèles testés",
        "modeling_text": """
        ### 🏆 Meilleur modèle : SVC avec PCA (81% de précision)
        - Excellente détection des radiographies **saines**
        - Bonne détection des cas de **pneumonie**
        - Plus difficile pour distinguer **virus** vs **bactérie**
        """,
        "model_images": [
            {"path": "../projets/zoidberg/images/svcwithpca.png", "caption": "Matrice de confusion"},
            {"path": "../projets/zoidberg/images/classification_rapport.png", "caption": "Rapport de classification"},
            {"path": "../projets/zoidberg/images/aucsvcwithpca.png", "caption": "Courbes ROC"},
        ],

        "comparison_title": "📊 Comparaison des autres modèles testés",
        "comparison_table": """
        | Modèle | PCA | F1-Score | Observations |
        |--------|-----|----------|--------------|
        | **SVC** | ✅ Oui | **79%** | 🏆 Meilleur équilibre |
        | SVC | ❌ Non | ~75% | Bonnes performances mais moins stable |
        | Random Forest | ✅ Oui | ~72% | Plus rapide mais moins précis |
        """,

        "test_model_title": "Tester le modèle",
        "Show_Proba": "Probabilités du modèle",
        "upload_label": "Uploader un scan pulmonaire (jpg/png)",
        "uploaded_image_caption": "Image importée",
        "healthy_label": "Sain",
        "bacterial_label": "Pneumonie bactérienne",
        "viral_label": "Pneumonie virale",
        "reset_button": "Réinitialiser la sélection",
        "info_uploaded": "Image uploadée par l’utilisateur",
        "correct_result": "✅ Le modèle a correctement identifié la catégorie : {expected_class}",
        "wrong_result": "❌ Erreur : le modèle a prédit {predicted_class} au lieu de {expected_class}"
    },

    "en": {
        "footer": "© 2025 - Personal portfolio built with Streamlit",
        "proj_zoidberg": "🩻 Lung Scan Classification",
        "proj_zoidberg_description": "**Goal:** predict whether a lung scan is healthy or shows pneumonia.",

        "metrics_title": "Performance Metrics of the Best Model",
        "metrics_rows": [
            [
                {"label": "Overall F1-Score", "value": "80%", "help": "The F1-score combines precision and recall to measure overall performance."},
                {"label": "Healthy (Normal)", "value": "92%"},
                {"label": "Bacterial Pneumonia", "value": "83%"},
                {"label": "Viral Pneumonia", "value": "62%"},
            ],
            [
                {"label": "Overall AUC", "value": "91.9%", "help": "AUC measures how well the model distinguishes between classes."},
                {"label": "Healthy (Normal)", "value": "98.6%"},
                {"label": "Bacterial Pneumonia", "value": "91.0%"},
                {"label": "Viral Pneumonia", "value": "86.1%"},
            ]
        ],

        "preprocess_data": "Data Preprocessing",
        "example_scans_title": "**Example Scans**",
        "example_scans": [
            {"caption": "Healthy Scan", "path": "../projets/zoidberg/images/Scan_Sain_Exemple.jpeg"},
            {"caption": "Bacterial Pneumonia", "path": "../projets/zoidberg/images/Scan_PneumonieBacterienne.jpeg"},
            {"caption": "Viral Pneumonia", "path": "../projets/zoidberg/images/Scan_PneumonieViral.jpeg"},
        ],

        "dataset_title": "**Dataset Statistics**",
        "dataset": {
            "Class": ["BACTERIA", "NORMAL", "VIRUS"],
            "Number of images": [2780, 1585, 1493],
            "Weight": [0.70, 1.23, 1.30]
        },
        "dataset_image_caption": "Class Distribution",

        "preprocessing_title": "**Cleaning and Preparation Steps**",
        "preprocessing_steps": [
            {"text": "- Normalization using StandardScaler to make scans comparable", "image": ["../projets/zoidberg/images/apresScaler.png", "../projets/zoidberg/images/avantScaler.png"], "caption": ""},
            {"text": "- Image resizing to standardize dimensions", "image": ["../projets/zoidberg/images/imageweight.png"], "caption": "Histogram of dimensions"},
            {"text": "- Data augmentation using horizontal flip (especially helpful for viral class)", "image": ["../projets/zoidberg/images/flipimage.png"], "caption": "Example of horizontal flip"},
        ],

        "models_title": "Tested Models",
        "modeling_text": """
        ### 🏆 Best Model: SVC with PCA (81% accuracy)
        - Excellent at detecting **healthy** X-rays
        - Good at identifying **pneumonia**
        - Harder to distinguish **virus** vs **bacteria**
        """,
        "model_images": [
            {"path": "../projets/zoidberg/images/svcwithpca.png", "caption": "Confusion Matrix"},
            {"path": "../projets/zoidberg/images/classification_rapport.png", "caption": "Classification Report"},
            {"path": "../projets/zoidberg/images/aucsvcwithpca.png", "caption": "ROC Curves"},
        ],

        "comparison_title": "📊 Comparison of Tested Models",
        "comparison_table": """
        | Model | PCA | F1-Score | Observations |
        |--------|-----|----------|--------------|
        | **SVC** | ✅ Yes | **79%** | 🏆 Best balance |
        | SVC | ❌ No | ~75% | Good performance but less stable |
        | Random Forest | ✅ Yes | ~72% | Faster but less accurate |
        """,

        "test_model_title": "Test the Model",
        "Show_Proba": "Model Probabilities",
        "upload_label": "Upload a lung scan (jpg/png)",
        "uploaded_image_caption": "Uploaded Image",
        "healthy_label": "Healthy",
        "bacterial_label": "Bacterial Pneumonia",
        "viral_label": "Viral Pneumonia",
        "reset_button": "Reset Selection",
        "info_uploaded": "Image uploaded by the user",
        "correct_result": "✅ The model correctly identified the category: {expected_class}",
        "wrong_result": "❌ Error: the model predicted {predicted_class} instead of {expected_class}"
    }
}
