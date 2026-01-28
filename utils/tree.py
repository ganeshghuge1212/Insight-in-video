def build_enhanced_tree(sections):
    """
    sections = [
        {
            "title": "Kafka",
            "points": ["Producer", "Consumer", "Broker"]
        }
    ]
    """

    tree = []

    for sec in sections:
        node = {
            "name": sec["title"],
            "children": []
        }

        for point in sec.get("points", []):
            node["children"].append({
                "name": point
            })

        tree.append(node)

    return tree
