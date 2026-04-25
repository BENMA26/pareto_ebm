#conflict list
CONFLICTS_DICT = {
    # 同组内“至多一个为真”
    "mutual_exclusion_groups": [
        # 头发颜色四选一
        [(8,  "Black_Hair"), (9,  "Blond_Hair"), (11, "Brown_Hair"), (17, "Gray_Hair")],
        # 发型形态二选一
        [(32, "Straight_Hair"), (33, "Wavy_Hair")],
    ],
    # 成对互斥（不可同时为真）
    "pairs": [
        # 秃头 vs 任意“有头发”特征
        [(4,  "Bald"), (5,  "Bangs")],
        [(4,  "Bald"), (8,  "Black_Hair")],
        [(4,  "Bald"), (9,  "Blond_Hair")],
        [(4,  "Bald"), (11, "Brown_Hair")],
        [(4,  "Bald"), (17, "Gray_Hair")],
        [(4,  "Bald"), (32, "Straight_Hair")],
        [(4,  "Bald"), (33, "Wavy_Hair")],
        # 无胡须 vs 各类胡须
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
    # 发色互斥（四选一，成对展开）
    [(8,  "Black_Hair"), (9,  "Blond_Hair")],
    [(8,  "Black_Hair"), (11, "Brown_Hair")],
    #[(8,  "Black_Hair"), (17, "Gray_Hair")],
    [(9,  "Blond_Hair"), (11, "Brown_Hair")],
    #[(9,  "Blond_Hair"), (17, "Gray_Hair")],
    #[(11, "Brown_Hair"), (17, "Gray_Hair")],

    # 发型互斥（二选一）
    [(32, "Straight_Hair"), (33, "Wavy_Hair")]
]

# attribute list
ATTRIBUTE_LIST = [
    ["5_o_Clock_Shadow", "五点钟胡渣"],
    ["Arched_Eyebrows", "弓形眉"],
    ["Attractive", "有吸引力"],
    ["Bags_Under_Eyes", "眼袋"],
    ["Bald", "秃头"],
    ["Bangs", "刘海"],
    ["Big_Lips", "厚嘴唇"],
    ["Big_Nose", "大鼻子"],
    ["Black_Hair", "黑发"],
    ["Blond_Hair", "金发"],
    ["Blurry", "模糊"],
    ["Brown_Hair", "棕发"],
    ["Bushy_Eyebrows", "浓眉"],
    ["Chubby", "微胖"],
    ["Double_Chin", "双下巴"],
    ["Eyeglasses", "戴眼镜"],
    ["Goatee", "山羊胡"],
    ["Gray_Hair", "灰白发"],
    ["Heavy_Makeup", "浓妆"],
    ["High_Cheekbones", "高颧骨"],
    ["Male", "男性"],
    ["Mouth_Slightly_Open", "嘴微张"],
    ["Mustache", "小胡子（上唇胡）"],
    ["Narrow_Eyes", "细长眼"],
    ["No_Beard", "无胡须"],
    ["Oval_Face", "椭圆脸"],
    ["Pale_Skin", "皮肤苍白"],
    ["Pointy_Nose", "尖鼻子"],
    ["Receding_Hairline", "发际线后移"],
    ["Rosy_Cheeks", "红润脸颊"],
    ["Sideburns", "鬓角"],
    ["Smiling", "微笑"],
    ["Straight_Hair", "直发"],
    ["Wavy_Hair", "波浪发"],
    ["Wearing_Earrings", "戴耳环"],
    ["Wearing_Hat", "戴帽子"],
    ["Wearing_Lipstick", "涂口红"],
    ["Wearing_Necklace", "戴项链"],
    ["Wearing_Necktie", "系领带"],
    ["Young", "年轻"],
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

