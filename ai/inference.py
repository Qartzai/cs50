# Simple code for inference by enumeration in a Bayesian network - Lecture 2
# Manual calculation approach (works without library dependencies)

# Define probability tables (nodes)
probability_tables = {
    "Rain": {
        "none": 0.7,
        "light": 0.2,
        "heavy": 0.1
    },
    "Maintenance | Rain": {
        ("none", "yes"): 0.4,
        ("none", "no"): 0.6,
        ("light", "yes"): 0.2,
        ("light", "no"): 0.8,
        ("heavy", "yes"): 0.1,
        ("heavy", "no"): 0.9
    },
    "Train | Rain, Maintenance": {
        ("none", "yes", "on time"): 0.8,
        ("none", "yes", "delayed"): 0.2,
        ("none", "no", "on time"): 0.9,
        ("none", "no", "delayed"): 0.1,
        ("light", "yes", "on time"): 0.6,
        ("light", "yes", "delayed"): 0.4,
        ("light", "no", "on time"): 0.7,
        ("light", "no", "delayed"): 0.3,
        ("heavy", "yes", "on time"): 0.4,
        ("heavy", "yes", "delayed"): 0.6,
        ("heavy", "no", "on time"): 0.5,
        ("heavy", "no", "delayed"): 0.5
    },
    "Appointment | Train": {
        ("on time", "attend"): 0.9,
        ("on time", "miss"): 0.1,
        ("delayed", "attend"): 0.6,
        ("delayed", "miss"): 0.4
    }
}

def calculate_joint_probability(rain, maintenance, train, appointment):
    """
    Calculate joint probability P(rain, maintenance, train, appointment)
    using the probability tables defined above.
    
    Bayesian Network structure:
    Rain -> Maintenance
    Rain -> Train
    Maintenance -> Train
    Train -> Appointment
    """
    
 
    p_rain = probability_tables["Rain"][rain]
    
    p_maintenance_given_rain = probability_tables["Maintenance | Rain"][(rain, maintenance)]
    
    p_train_given_rain_maintenance = probability_tables["Train | Rain, Maintenance"][(rain, maintenance, train)]

    p_appointment_given_train = probability_tables["Appointment | Train"][(train, appointment)]

    return p_rain * p_maintenance_given_rain * p_train_given_rain_maintenance * p_appointment_given_train
    

# Calculate the probability
probability = calculate_joint_probability("heavy", "no", "delayed", "attend")

print(f"Bayesian Network Inference")
print(f"=" * 50)
print(f"Query: P(Rain=heavy, Maintenance=no, Train=delayed, Appointment=attend)")
print(f"Result: {probability:.6f}")
print(f"\nCalculation:")
print(f"  P(Rain=heavy) = {probability_tables['Rain']['heavy']}")
print(f"  P(Maintenance=no | Rain=heavy) = {probability_tables['Maintenance | Rain'][('heavy', 'no')]}")
print(f"  P(Train=delayed | Rain=heavy, Maintenance=no) = {probability_tables['Train | Rain, Maintenance'][('heavy', 'no', 'delayed')]}")
print(f"  P(Appointment=attend | Train=delayed) = {probability_tables['Appointment | Train'][('delayed', 'attend')]}")
print(f"  Joint Probability = {probability_tables['Rain']['heavy']} × {probability_tables['Maintenance | Rain'][('heavy', 'no')]} × {probability_tables['Train | Rain, Maintenance'][('heavy', 'no', 'delayed')]} × {probability_tables['Appointment | Train'][('delayed', 'attend')]} = {probability:.6f}")