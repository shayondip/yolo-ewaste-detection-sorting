SORTING_RULES = {
    "Battery_Waste": {
        "category": "Battery / E-Waste",
        "bin": "Battery Collection Bin",
        "instruction": "Keep separate and send to an authorized battery/e-waste collection point."
    },
    "Glass_Waste": {
        "category": "Glass",
        "bin": "Glass Waste Bin",
        "instruction": "Place in the designated glass recycling stream."
    },
    "Keyboard": {
        "category": "Electronics",
        "bin": "E-Waste / Electronics Bin",
        "instruction": "Send to an electronics recycling or e-waste collection stream."
    },
    "Light_Bulb": {
        "category": "Special / Lamp Waste",
        "bin": "Special Handling Bin",
        "instruction": "Handle separately; use an appropriate lamp/light-bulb collection stream."
    },
    "Medical_Waste": {
        "category": "Medical Waste",
        "bin": "Medical Waste Bin",
        "instruction": "Do not mix with household waste; use the appropriate medical-waste disposal stream."
    },
    "Metal_Waste": {
        "category": "Metal",
        "bin": "Metal Recycling Bin",
        "instruction": "Send to the designated metal recycling stream."
    },
    "Mobile": {
        "category": "Electronics",
        "bin": "E-Waste / Electronics Bin",
        "instruction": "Send to an electronics recycling or e-waste collection stream."
    },
    "Mouse": {
        "category": "Electronics",
        "bin": "E-Waste / Electronics Bin",
        "instruction": "Send to an electronics recycling or e-waste collection stream."
    },
    "Organic_Waste": {
        "category": "Organic Waste",
        "bin": "Organic Waste Bin",
        "instruction": "Place in the organic/compostable waste stream where available."
    },
    "PCB": {
        "category": "Electronic PCB",
        "bin": "PCB / E-Waste Bin",
        "instruction": "Send to an authorized electronic-waste recycling stream."
    },
    "Paper_Waste": {
        "category": "Paper",
        "bin": "Paper Recycling Bin",
        "instruction": "Place in the designated paper recycling stream."
    },
    "Plastic_Waste": {
        "category": "Plastic",
        "bin": "Plastic Recycling Bin",
        "instruction": "Place in the designated plastic recycling stream."
    },
}


def get_sorting_recommendation(class_name: str) -> dict:
    """Return sorting information for a detected class."""
    return SORTING_RULES.get(
        class_name,
        {
            "category": "Unknown",
            "bin": "Manual Review",
            "instruction": "No sorting rule is configured for this class."
        },
    )