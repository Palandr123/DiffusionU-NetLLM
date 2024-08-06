spot_object_template = [
    {
        "role": "user",
        "content": "User prompt: A brown horse is beneath a black dog. Another orange cat is beneath a brown horse.",
    },
    {
        "role": "assistant",
        "content": """Reasoning: The description talks about three objects: a brown horse, a black dog, and an orange cat. We report the color attribute thoroughly. No specified negation terms. No background is mentioned and thus fill in the default one.
    Objects: [('horse', ['brown']), ('dog', ['black']), ('cat', ['orange'])]
    Background: A realistic image
    Negation: """,
    },
    {
        "role": "user",
        "content": "User prompt: There's a white car and a yellow airplane in a garage. They're in front of two dogs and behind a cat. The car is small. Another yellow car is outside the garage.",
    },
    {
        "role": "assistant",
        "content": """Reasoning: The scene has two cars, one airplane, two dogs, and a cat. The car and airplane have colors. The first car also has a size. No specified negation terms. The background is a garage.
    Objects: [('car', ['white and small', 'yellow']), ('airplane', ['yellow']), ('dog', [None, None]), ('cat', [None])]
    Background: A realistic image in a garage
    Negation: """,
    },
    {
        "role": "user",
        "content": "User prompt: A car and a dog are on top of an airplane and below a red chair. There's another dog sitting on the mentioned chair.",
    },
    {
        "role": "assistant",
        "content": """Reasoning: Four objects are described: one car, airplane, two dog, and a chair. The chair is red color. No specified negation terms. No background is mentioned and thus fill in the default one.
    Objects: [('car', [None]), ('airplane', [None]), ('dog', [None, None]), ('chair', ['red'])]
    Background: A realistic image
    Negation: """,
    },
    {
        "role": "user",
        "content": "User prompt: An oil painting at the beach of a blue bicycle to the left of a bench and to the right of a palm tree with five seagulls in the sky.",
    },
    {
        "role": "assistant",
        "content": """Reasoning: Here, there are five seagulls, one blue bicycle, one palm tree, and one bench. No specified negation terms. The background is an oil painting at the beach.
    Objects: [('bicycle', ['blue']), ('palm tree', [None]), ('seagull', [None, None, None, None, None]), ('bench', [None])]
    Background: An oil painting at the beach
    Negation: """,
    },
    {
        "role": "user",
        "content": "User prompt: An animated-style image of a scene without backpacks.",
    },
    {
        "role": "assistant",
        "content": """Reasoning: The description clearly states no backpacks, so this must be acknowledged. The user provides the negative prompt of backpacks. The background is an animated-style image.
    Objects: [('backpacks', [None])]
    Background: An animated-style image
    Negation: backpacks""",
    },
    {
        "role": "user",
        "content": "User Prompt: Make the dog a sleeping dog and remove all shadows in an image of a grassland.",
    },
    {
        "role": "assistant",
        "content": """Reasoning: The user prompt specifies a sleeping dog on the image and a shadow to be removed. The background is a realistic image of a grassland.                                                                                                                              
    Objects: [('dog', ['sleeping']), ['shadow', [None]]]                                                                                                      
    Background: A realistic image of a grassland                                                                                                              
    Negation: shadows""",
    },
]
