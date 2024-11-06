from transformers import pipeline



def generer_legende_en_francais(chemin_image):
    """
    Génère une légende pour une image donnée en anglais et la traduit en français.

    :param chemin_image: Le chemin vers l'image pour laquelle générer une légende.
    :type chemin_image: str
    :return: La légende traduite en français.
    :rtype: str
    """

    # Initialiser la pipeline de légendage d'image
    pipeline_legende_image = pipeline("image-to-text", model="Salesforce/blip-image-captioning-large")

    # Initialiser la pipeline de traduction de l'anglais vers le français
    pipeline_traduction = pipeline("translation", model="Helsinki-NLP/opus-mt-en-fr")

    # Génération de la légende en anglais
    resultats_legende = pipeline_legende_image(chemin_image)
    legende_anglaise = resultats_legende[0]['generated_text']

    # Traduction de la légende en français
    legende_francaise = pipeline_traduction(legende_anglaise, max_length=512)
    texte_francais = legende_francaise[0]['translation_text']

    return texte_francais

