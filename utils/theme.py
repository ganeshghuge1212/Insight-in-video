def get_theme(theme_name="dark"):
    if theme_name == "light":
        return {
            "bg": "#ffffff",
            "text": "#000000",
            "card": "#f5f5f5"
        }

    return {
        "bg": "#0e1117",
        "text": "#ffffff",
        "card": "#1f2933"
    }
