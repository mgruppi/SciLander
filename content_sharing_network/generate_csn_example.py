"""
This script generates an example of a CSN network for illustrative purposes
"""

from networkx import DiGraph, write_gml

g = DiGraph()

sources = [
    {
        "name": "Washington Post",
        "class": 0
    },
    {
        "name": "Charlotte Observer",
        "class": 0
    },
    {
        "name": "US News",
        "class": 0
    },
    {
        "name": "CBS News",
        "class": 0
    },
    {
        "name": "Raw Story",
        "class": 0
    },
    {
        "name": "Veterans Today",
        "class": 1
    },
    {
        "name": "Global Research",
        "class": 1
    },
    {
        "name": "The Greanville Post",
        "class": 1
    },
    {
        "name": "Russophile",
        "class": 1
    },
    {
        "name": "Breitbart",
        "class": 1
    },
    {
        "name": "The Epoch Times",
        "class": 1
    },
    {
        "name": "What Really Happened",
        "class": 1
    }
]

for i, s in enumerate(sources):
    g.add_node(s["name"], **s)

edges = [
    ("Washington Post", "Charlotte Observer", 0.75),
    ("Washington Post", "US News", 0.5),
    ("Washington Post", "CBS News", 0.2),
    ("CBS News", "Charlotte Observer", 0.3),
    ("Raw Story", "CBS News", 0.25),
    ("US News", "CBS News", 0.6),

    ("Washington Post", "Breitbart", 0.2),
    ("Raw Story", "Breitbart", 0.1),
    ("Raw Story", "Global Research", 0.15),
    ("Raw Story", "Russophile", 0.01),
    ("Raw Story", "What Really Happened", 0.2),
    ("Washington Post", "What Really Happened", 0.18),
    ("US News", "What Really Happened", 0.2),

    ("Veterans Today", "Global Research", 0.75),
    ("Veterans Today", "The Greanville Post", 0.6),
    ("Veterans Today", "Russophile", 0.8),
    ("Veterans Today", "Breitbart", 0.5),
    ("Veterans Today", "What Really Happened", 0.5),
    ("Breitbart", "The Greanville Post", 0.5),
    ("Breitbart", "Russophile", 0.6),
    ("Breitbart", "What Really Happened", 0.7),
    ("The Epoch Times", "Russophile", 0.6),
    ("The Epoch Times", "Breitbart", 0.25),
    ("The Epoch Times", "Veterans Today", 0.4),
    ("The Epoch Times", "What Really Happened", 0.3),
    ("Global Research", "Russophile", 0.3),
    ("Global Research", "What Really Happened", 0.25)  
]

g.add_weighted_edges_from(edges)

write_gml(g, "csn_example.gml")
