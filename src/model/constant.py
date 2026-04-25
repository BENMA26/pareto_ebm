#conflict list
CONFLICTS_DICT = {
    # At most one can be true within each group.
    "mutual_exclusion_groups": [
        # Hair color: choose one out of four.
        [(8,  "Black_Hair"), (9,  "Blond_Hair"), (11, "Brown_Hair"), (17, "Gray_Hair")],
        # Hair style: choose one out of two.
        [(32, "Straight_Hair"), (33, "Wavy_Hair")],
    ],
    # Pairwise conflicts (cannot be true at the same time).
    "pairs": [
        # Bald vs any "has hair" attributes.
        [(4,  "Bald"), (5,  "Bangs")],
        [(4,  "Bald"), (8,  "Black_Hair")],
        [(4,  "Bald"), (9,  "Blond_Hair")],
        [(4,  "Bald"), (11, "Brown_Hair")],
        [(4,  "Bald"), (17, "Gray_Hair")],
        [(4,  "Bald"), (32, "Straight_Hair")],
        [(4,  "Bald"), (33, "Wavy_Hair")],
        # No_Beard vs facial-hair attributes.
        [(24, "No_Beard"), (0,  "5_o_Clock_Shadow")],
        [(24, "No_Beard"), (22, "Mustache")],
        [(24, "No_Beard"), (16, "Goatee")],
        [(24, "No_Beard"), (30, "Sideburns")],
        [(4,  "Bald"), (28, "Receding_Hairline")],
        [(5,  "Bangs"), (28, "Receding_Hairline")],
        [(39, "Young"), (17, "Gray_Hair")],
    ],
}

CONFLICTS_LIST = [
    # Hair color conflicts (one-of-four, expanded pairwise).
    [(8,  "Black_Hair"), (9,  "Blond_Hair")],
    [(8,  "Black_Hair"), (11, "Brown_Hair")],
    #[(8,  "Black_Hair"), (17, "Gray_Hair")],
    [(9,  "Blond_Hair"), (11, "Brown_Hair")],
    #[(9,  "Blond_Hair"), (17, "Gray_Hair")],
    #[(11, "Brown_Hair"), (17, "Gray_Hair")],

    # Hair style conflicts (one-of-two).
    [(32, "Straight_Hair"), (33, "Wavy_Hair")]
]

# attribute list
ATTRIBUTE_LIST = [
    ["5_o_Clock_Shadow", "Five o'clock shadow"],
    ["Arched_Eyebrows", "Arched eyebrows"],
    ["Attractive", "Attractive"],
    ["Bags_Under_Eyes", "Bags under eyes"],
    ["Bald", "Bald"],
    ["Bangs", "Bangs"],
    ["Big_Lips", "Big lips"],
    ["Big_Nose", "Big nose"],
    ["Black_Hair", "Black hair"],
    ["Blond_Hair", "Blond hair"],
    ["Blurry", "Blurry"],
    ["Brown_Hair", "Brown hair"],
    ["Bushy_Eyebrows", "Bushy eyebrows"],
    ["Chubby", "Chubby"],
    ["Double_Chin", "Double chin"],
    ["Eyeglasses", "Eyeglasses"],
    ["Goatee", "Goatee"],
    ["Gray_Hair", "Gray hair"],
    ["Heavy_Makeup", "Heavy makeup"],
    ["High_Cheekbones", "High cheekbones"],
    ["Male", "Male"],
    ["Mouth_Slightly_Open", "Mouth slightly open"],
    ["Mustache", "Mustache"],
    ["Narrow_Eyes", "Narrow eyes"],
    ["No_Beard", "No beard"],
    ["Oval_Face", "Oval face"],
    ["Pale_Skin", "Pale skin"],
    ["Pointy_Nose", "Pointy nose"],
    ["Receding_Hairline", "Receding hairline"],
    ["Rosy_Cheeks", "Rosy cheeks"],
    ["Sideburns", "Sideburns"],
    ["Smiling", "Smiling"],
    ["Straight_Hair", "Straight hair"],
    ["Wavy_Hair", "Wavy hair"],
    ["Wearing_Earrings", "Wearing earrings"],
    ["Wearing_Hat", "Wearing hat"],
    ["Wearing_Lipstick", "Wearing lipstick"],
    ["Wearing_Necklace", "Wearing necklace"],
    ["Wearing_Necktie", "Wearing necktie"],
    ["Young", "Young"],
]

ATTRIBUTE_NAME_LIST = [
    "5_o_Clock_Shadow",
    "Arched_Eyebrows",
    "Attractive",
    "Bags_Under_Eyes",
    "Bald",
    "Bangs",
    "Big_Lips",
    "Big_Nose",
    "Black_Hair",
    "Blurry",
    "Brown_Hair",
    "Bushy_Eyebrows",
    "Chubby",
    "Double_Chin",
    "Eyeglasses",
    "Goatee",
    "Gray_Hair",
    "Heavy_Makeup",
    "High_Cheekbones",
    "Male",
    "Mouth_Slightly_Open",
    "Mustache",
    "Narrow_Eyes",
    "No_Beard",
    "Oval_Face",
    "Pale_Skin",
    "Pointy_Nose",
    "Receding_Hairline",
    "Rosy_Cheeks",
    "Sideburns",
    "Smiling",
    "Straight_Hair",
    "Wavy_Hair",
    "Wearing_Earrings",
    "Wearing_Hat",
    "Wearing_Lipstick",
    "Wearing_Necklace",
    "Wearing_Necktie",
    "Young",
    "Blond_Hair"
]
