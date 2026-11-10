"""Langue de l'interface : français (défaut) ou anglais.

Les textes français servent de clés ; leur traduction anglaise est dans i18n_en.py.
Un texte absent du dictionnaire reste en français.
"""
from .i18n_en import EN

_lang = "fr"


def set_language(lang: str) -> None:
    global _lang
    _lang = "en" if lang == "en" else "fr"


def language() -> str:
    return _lang


def t(text: str) -> str:
    """Texte traduit dans la langue courante."""
    return EN.get(text, text) if _lang == "en" else text
