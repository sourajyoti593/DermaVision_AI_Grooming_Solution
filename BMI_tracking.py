def calculate_bmi(weight, height):
    """Calculate BMI"""
    bmi = weight / (height ** 2)
    return round(bmi, 2)

def calculate_body_fat(bmi, age, gender):
    """Calculate Body Fat Percentage using BMI"""
    if gender.lower() == "male":
        bf_percentage = (1.20 * bmi) + (0.23 * age) - 16.2
    else:
        bf_percentage = (1.20 * bmi) + (0.23 * age) - 5.4
    return round(bf_percentage, 2)

def calculate_muscle_mass(body_fat_percentage):
    """Estimate Muscle Mass Percentage (Rule-based)"""
    muscle_mass = 100 - body_fat_percentage - 15 - 5  # Bone Mass ~15%, Other Tissues ~5%
    return round(muscle_mass, 2)

def calculate_ideal_weight(height, gender):
    """Calculate Ideal Weight using Devine Formula"""
    height_inches = height * 39.3701  # Convert meters to inches
    if gender.lower() == "male":
        ideal_weight = 50 + (2.3 * (height_inches - 60))
    else:
        ideal_weight = 45.5 + (2.3 * (height_inches - 60))
    return round(ideal_weight, 2)

# --- User Input ---
weight = float(input("Enter your weight (kg): "))
height = float(input("Enter your height (m): "))
age = int(input("Enter your age: "))
gender = input("Enter your gender (male/female): ")

# --- Calculations ---
bmi = calculate_bmi(weight, height)
body_fat_percentage = calculate_body_fat(bmi, age, gender)
muscle_mass_percentage = calculate_muscle_mass(body_fat_percentage)
ideal_weight = calculate_ideal_weight(height, gender)

# --- Display Results ---
print("\n--- Health Analysis ---")
print(f"BMI: {bmi}")
print(f"Body Fat Percentage: {body_fat_percentage}%")
print(f"Muscle Mass Percentage: {muscle_mass_percentage}%")
print(f"Ideal Weight for Your Height: {ideal_weight} kg")

# --- Suggestion ---
if weight < ideal_weight - 5:
    print("💡 Suggestion: You may need to gain weight for a healthy balance.")
elif weight > ideal_weight + 5:
    print("💡 Suggestion: You may need to lose weight for optimal health.")
else:
    print("✅ Your weight is within a healthy range!")
